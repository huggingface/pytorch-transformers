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
Processor class for PaliGemma.
"""

from typing import List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, is_valid_image
from ...processing_utils import (
    ImagesKwargs,
    ProcessingKwargs,
    ProcessorMixin,
    TextKwargs,
    Unpack,
    _validate_images_text_input_order,
)
from ...tokenization_utils_base import (
    AddedToken,
    PreTokenizedInput,
    TextInput,
)
from ...utils import logging


logger = logging.get_logger(__name__)

IMAGE_TOKEN = "<image>"
EXTRA_TOKENS = [f"<loc{i:0>4}>" for i in range(1024)] + [f"<seg{i:0>3}>" for i in range(128)]


class PaliGemmaTextKwargs(TextKwargs):
    suffix: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]]


class PaliGemmaImagesKwargs(ImagesKwargs):
    do_convert_rgb: Optional[bool]


class PaliGemmaProcessorKwargs(ProcessingKwargs, total=False):
    text_kwargs: PaliGemmaTextKwargs
    images_kwargs: PaliGemmaImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
        "images_kwargs": {
            "data_format": "channels_first",
        },
    }


# Copied from transformers.models.idefics2.processing_idefics2.is_url
def is_url(val) -> bool:
    return isinstance(val, str) and val.startswith("http")


# Copied from transformers.models.idefics2.processing_idefics2.is_image_or_image_url
def is_image_or_image_url(elem):
    return is_url(elem) or is_valid_image(elem)


def _is_str_or_image(elem):
    return isinstance(elem, (str)) or is_image_or_image_url(elem)


def build_string_from_input(prompt, bos_token, image_seq_len, image_token, num_images):
    """
    Builds a string from the input prompt and image tokens.
    For example, for the call:
    build_string_from_input(
        prompt="Prefix str"
        bos_token="<s>",
        image_seq_len=3,
        image_token="<im>",
    )
    The output will be:
    "<im><im><im><s>Initial str"
    Args:
        prompt (`List[Union[str, ImageInput]]`): The input prompt.
        bos_token (`str`): The beginning of sentence token.
        image_seq_len (`int`): The length of the image sequence.
        image_token (`str`): The image token.
        num_images (`int`): Number of images in the prompt.
    """
    return f"{image_token * image_seq_len * num_images}{bos_token}{prompt}\n"


# Copied from transformers.models.llava_next.image_processing_llava_next.make_batched_images
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

    elif is_valid_image(images):
        return [images]

    raise ValueError(f"Could not make batched video from {images}")


class PaliGemmaProcessor(ProcessorMixin):
    r"""
    Constructs a PaliGemma processor which wraps a PaliGemma image processor and a PaliGemma tokenizer into a single processor.

    [`PaliGemmaProcessor`] offers all the functionalities of [`SiglipImageProcessor`] and [`GemmaTokenizerFast`]. See the
    [`~PaliGemmaProcessor.__call__`] and [`~PaliGemmaProcessor.decode`] for more information.

    Args:
        image_processor ([`SiglipImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`GemmaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "SiglipImageProcessor"
    tokenizer_class = ("GemmaTokenizer", "GemmaTokenizerFast")

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        chat_template=None,
        **kwargs,
    ):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        if not hasattr(image_processor, "image_seq_length"):
            raise ValueError("Image processor is missing an `image_seq_length` attribute.")

        self.image_seq_length = image_processor.image_seq_length

        if not hasattr(tokenizer, "image_token"):
            image_token = AddedToken(IMAGE_TOKEN, normalized=False, special=True)
            tokens_to_add = {"additional_special_tokens": [image_token]}
            tokenizer.add_special_tokens(tokens_to_add)
            self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        else:
            self.image_token_id = tokenizer.image_token_id

        tokenizer.add_tokens(EXTRA_TOKENS)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: ImageInput = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[PaliGemmaProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to GemmaTokenizerFast's [`~GemmaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        SiglipImageProcessor's [`~SiglipImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        The usage for PaliGemma fine-tuning preparation is slightly different than usual. suffix passed are suffixes to
        the prompt in `text`, and will be placed after the prompt. This is because attention is handled differently for
        the prefix and the suffix. For instance,
        ```python
        image = PIL_cow_image
        prompt = "answer en Where is the cow standing?"
        suffix = "on the beach"
        inputs = processor(text=prompt, images=image, suffix=suffix)
        ```
        Here `inputs` will contain the `input_ids` and `token_type_ids` that follow
        ```python
        inputs["input_ids"][:, 256:]
        # tensor([[     2,   6006,    603,    573,  13910,   9980, 235336,    108,    477,   573,   8318]])
        inputs["token_type_ids"][:, 256:]
        tensor([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])
        ```
        Meaning the last three tokens are of "label" ("suffix") type while the other ones are of "prefix" type.


        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
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
            suffix (`str`, `List[str]`, `List[List[str]]`):
                The suffixes or batch of suffixes to be encoded. Only necessary for finetuning. See https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md
                for more information. If your prompt is "<image> What is on the image", the suffix corresponds to the expected prediction "a cow sitting on a bench".

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`. If `suffix`
              is provided, the `input_ids` will also contain the suffix input ids.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **labels** -- Labels compatible with training if `suffix` is not None
        """
        # check if images and text inputs are reversed for BC
        images, text = _validate_images_text_input_order(images, text)

        output_kwargs = self._merge_kwargs(
            PaliGemmaProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )
        suffix = output_kwargs["text_kwargs"].pop("suffix", None)

        return_token_type_ids = True if suffix is not None else False

        if images is None:
            raise ValueError("`images` are expected as arguments to a `PaliGemmaProcessor` instance.")
        if text is None:
            logger.warning_once(
                "You are using PaliGemma without a text prefix. It will perform as a picture-captioning model."
            )
            text = ""

        if _is_str_or_image(text):
            text = [text]
        elif isinstance(text, list) and _is_str_or_image(text[0]):
            pass

        if text is not None and images is not None:
            if not any(IMAGE_TOKEN in sample for sample in text):
                logger.warning(
                    "You are passing both `text` and `images` to `PaliGemmaProcessor`. The processor expects special "
                    "image tokens in the text, as many tokens as there are images per each text. It is recommended to "
                    "add `<image>` tokens in the very beginning of your text. For this call, we will infer how many images "
                    "each text has and add special tokens."
                )

                if isinstance(text, List) and isinstance(images, List):
                    if len(images) != len(text):
                        raise ValueError(
                            f"Received {len(images)} images for {len(text)} prompts. Each prompt should be associated with an image or list of images."
                        )

                # make a nested list of lists to be able to iterate over the images and text below
                if is_valid_image(images):
                    images = [[images]]
                elif isinstance(images, list) and is_valid_image(images[0]):
                    images = [[image] for image in images]
                elif not (isinstance(images, list) and isinstance(images[0], list) and is_valid_image(images[0][0])):
                    raise ValueError("images must be an image, list of images or list of list of images")

                input_strings = [
                    build_string_from_input(
                        prompt=prompt,
                        bos_token=self.tokenizer.bos_token,
                        image_seq_len=self.image_seq_length,
                        image_token=IMAGE_TOKEN,
                        num_images=len(image_list) if isinstance(image_list, list) else 1,
                    )
                    for prompt, image_list in zip(text, images)
                ]
                images = make_batched_images(images)
            else:
                expanded_samples = []
                for sample in text:
                    expanded_sample = sample.replace(IMAGE_TOKEN, IMAGE_TOKEN * self.image_seq_length)
                    bos_rfind_index = expanded_sample.rfind(IMAGE_TOKEN)
                    bos_index = bos_rfind_index + len(IMAGE_TOKEN) if bos_rfind_index != -1 else 0
                    expanded_sample = (
                        expanded_sample[:bos_index] + self.tokenizer.bos_token + expanded_sample[bos_index:]
                    )
                    expanded_samples.append(expanded_sample)
                input_strings = [f"{sample}\n" for sample in expanded_samples]

        if suffix is not None and _is_str_or_image(suffix):
            suffix = [suffix]
        if suffix is not None:
            suffix = [sfx + self.tokenizer.eos_token for sfx in suffix]
        pixel_values = self.image_processor(images, **output_kwargs["images_kwargs"])["pixel_values"]

        # max_length has to account for the image tokens
        if output_kwargs["text_kwargs"].get("max_length", None) is not None:
            output_kwargs["text_kwargs"]["max_length"] += self.image_seq_length

        inputs = self.tokenizer(
            input_strings,
            text_pair=suffix,
            return_token_type_ids=return_token_type_ids,
            **output_kwargs["text_kwargs"],
        )

        return_data = {**inputs, "pixel_values": pixel_values}

        if return_token_type_ids:
            labels = inputs["input_ids"].masked_fill(inputs["token_type_ids"] == 0, -100)
            return_data.update({"labels": labels})
        return BatchFeature(data=return_data)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Gemma
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Gemma
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to GemmaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names with CLIP->PaliGemma
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["PaliGemmaProcessor"]
