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

import enum
from typing import Dict, List, Optional, Union

from ..processing_utils import ProcessingKwargs, Unpack
from ..utils import (
    add_end_docstrings,
    is_torch_available,
    is_vision_available,
    logging,
    requires_backends,
)
from .base import Pipeline, build_pipeline_init_args


if is_vision_available():
    from PIL import Image

    from ..image_utils import load_images, valid_images


if is_torch_available():
    from ..models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
    from .pt_utils import KeyDataset

logger = logging.get_logger(__name__)

IMAGE_TOKEN = "<image>"


class ReturnType(enum.Enum):
    TENSORS = 0
    NEW_TEXT = 1
    FULL_TEXT = 2


class Chat:
    """This class is intended to just be used internally in this pipeline and not exposed to users. We convert chats
    to this format because the rest of the pipeline code tends to assume that lists of messages are
    actually a batch of samples rather than messages in the same conversation."""

    def __init__(self, messages: Dict, images: Union[str, List[str], "Image.Image", List["Image.Image"]]):
        for message in messages:
            if not ("role" in message and "content" in message):
                raise ValueError("When passing chat dicts as input, each dict must have a 'role' and 'content' key.")
        images = retrieve_images_in_messages(messages, images)

        self.messages = messages
        self.images = images


def retrieve_images_in_messages(
    messages: dict, images: Optional[Union[str, List[str], "Image.Image", List["Image.Image"]]]
):
    """
    Retrieve and combine images from the chat and the images passed as input.
    """
    if images is None:
        images = []
    idx_images = 0
    retrieved_images = []
    for message in messages:
        for content in message["content"]:
            if isinstance(content, dict):
                if content.get("type") == "image":
                    for key in ["image", "url", "path", "base64"]:
                        if key in content:
                            retrieved_images.append(content[key])
                            break
                    else:
                        if idx_images < len(images):
                            retrieved_images.append(images[idx_images])
                            idx_images += 1
                        else:
                            raise ValueError(
                                "The number of images in the chat messages should be the same as the number of images passed to the pipeline."
                            )
                # Add support for OpenAI/TGI chat format
                elif content.get("type") == "image_url":
                    if isinstance(content.get("image_url"), dict) and "url" in content["image_url"]:
                        retrieved_images.append(content["image_url"]["url"])
                        # Rewrite content to be in the Transformers chat format
                        content["type"] = "image"
                        content["image"] = content["image_url"]["url"]
                        del content["image_url"]
                    else:
                        raise ValueError(
                            "Wrong format for 'image_url' content type. The content should have an 'image_url' dict with a 'url' key."
                        )

    # The number of images passed should be consistent with the number of images in the chat without an image key
    if idx_images != len(images):
        raise ValueError(
            "The number of images in the chat messages should be the same as the number of images passed to the pipeline."
        )

    return retrieved_images


@add_end_docstrings(build_pipeline_init_args(has_processor=True))
class ImageTextToTextPipeline(Pipeline):
    """
    Image-text-to-text pipeline using an `AutoModelForImageTextToText`. This pipeline generates text given an image and text.
    When the underlying model is a conversational model, it can also accept one or more chats,
    in which case the pipeline will operate in chat mode and will continue the chat(s) by adding its response(s).
    Each chat takes the form of a list of dicts, where each dict contains "role" and "content" keys.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> pipe = pipeline(task="image-text-to-text", model="Salesforce/blip-image-captioning-base")
    >>> pipe("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png", text="A photo of")
    [{'generated_text': 'a photo of two birds'}]
    ```

    ```python
    >>> from transformers import pipeline

    >>> pipe = pipeline("image-text-to-text", model="llava-hf/llava-interleave-qwen-0.5b-hf")
    >>> messages = [
    >>>     {
    >>>         "role": "user",
    >>>         "content": [
    >>>             {
    >>>                 "type": "image",
    >>>                 "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
    >>>             },
    >>>             {"type": "text", "text": "Describe this image."},
    >>>         ],
    >>>     },
    >>>     {
    >>>         "role": "assistant",
    >>>         "content": [
    >>>             {"type": "text", "text": "There is a dog and"},
    >>>         ],
    >>>     },
    >>> ]
    >>> pipe(text=messages, max_new_tokens=20, return_full_text=False)
    [{'input_text': [{'role': 'user',
        'content': [{'type': 'image',
        'url': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
        {'type': 'text', 'text': 'Describe this image.'}]},
    {'role': 'assistant',
        'content': [{'type': 'text', 'text': 'There is a dog and'}]}],
    'generated_text': ' a person in the image. The dog is sitting on the sand, and the person is sitting on'}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This image-text to text pipeline can currently be loaded from pipeline() using the following task identifier:
    "image-text-to-text".

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?pipeline_tag=image-text-to-text).
    """

    _load_processor = True
    _load_image_processor = False
    _load_feature_extractor = False
    _load_tokenizer = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        requires_backends(self, "vision")
        self.check_model_type(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES)

    def _sanitize_parameters(
        self,
        max_new_tokens=None,
        generate_kwargs=None,
        timeout=None,
        return_full_text=None,
        return_tensors=None,
        return_type=None,
        continue_final_message=None,
        **kwargs: Unpack[ProcessingKwargs],
    ):
        forward_kwargs = {}
        preprocess_params = {}
        postprocess_params = {}

        preprocess_params["processing_kwargs"] = kwargs

        if timeout is not None:
            preprocess_params["timeout"] = timeout

        if continue_final_message is not None:
            preprocess_params["continue_final_message"] = continue_final_message

        if generate_kwargs is not None:
            forward_kwargs["generate_kwargs"] = generate_kwargs

        if max_new_tokens is not None:
            if "generate_kwargs" not in forward_kwargs:
                forward_kwargs["generate_kwargs"] = {}
            if "max_new_tokens" in forward_kwargs["generate_kwargs"]:
                raise ValueError(
                    "'max_new_tokens' is defined twice, once in 'generate_kwargs' and once as a direct parameter,"
                    " please use only one"
                )
            forward_kwargs["generate_kwargs"]["max_new_tokens"] = max_new_tokens

        if return_full_text is not None and return_type is None:
            if return_tensors is not None:
                raise ValueError("`return_full_text` is mutually exclusive with `return_tensors`")
            return_type = ReturnType.FULL_TEXT if return_full_text else ReturnType.NEW_TEXT
        if return_tensors is not None and return_type is None:
            return_type = ReturnType.TENSORS
        if return_type is not None:
            postprocess_params["return_type"] = return_type
        if continue_final_message is not None:
            postprocess_params["continue_final_message"] = continue_final_message

        return preprocess_params, forward_kwargs, postprocess_params

    def __call__(
        self,
        images: Optional[
            Union[str, List[str], List[List[str]], "Image.Image", List["Image.Image"], List[List["Image.Image"]]]
        ] = None,
        text: Optional[Union[str, List[str], List[dict]]] = None,
        **kwargs,
    ):
        """
        Generate a text given text and the image(s) passed as inputs.

        Args:
            images (`str`, `List[str]`, `PIL.Image or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a HTTP(s) link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images.
            text (str, List[str], `List[Dict[str, Union[str, PIL.Image]]]`):
                The text to be used for generation. If a list of strings is passed, the length of the list should be the
                same as the number of images. Text can also follow the chat format: a list of dictionaries where each
                dictionary represents a message in a conversation. Each dictionary should have two keys: 'role' and
                'content'. 'role' should be one of 'user', 'system' or 'assistant'. 'content' should be a list of dictionary
                containing the text of the message and the type of the message. The type of the message can be either
                'text' or 'image'. If the type is 'image', no text is needed.
            return_tensors (`bool`, *optional*, defaults to `False`):
                Returns the tensors of predictions (as token indices) in the outputs. If set to
                `True`, the decoded text is not returned.
            return_text (`bool`, *optional*):
                Returns the decoded texts in the outputs.
            return_full_text (`bool`, *optional*, defaults to `True`):
                If set to `False` only added text is returned, otherwise the full text is returned. Cannot be
                specified at the same time as `return_text`.
            continue_final_message( `bool`, *optional*): This indicates that you want the model to continue the
                last message in the input chat rather than starting a new one, allowing you to "prefill" its response.
                By default this is `True` when the final message in the input chat has the `assistant` role and
                `False` otherwise, but you can manually override that behaviour by setting this flag.

        Return:
            A list or a list of list of `dict`: Each result comes as a dictionary with the following key (cannot return a combination
            of both `generated_text` and `generated_token_ids`):

            - **generated_text** (`str`, present when `return_text=True`) -- The generated text.
            - **generated_token_ids** (`torch.Tensor`, present when `return_tensors=True`) -- The token
                ids of the generated text.
            - **input_text** (`str`) -- The input text.
        """
        if images is None and text is None:
            raise ValueError("You must at least provide either text or images.")
        if images is not None and text is None and not valid_images(images):
            """
            Supports the following format
            - {"image": image, "text": text}
            - [{"image": image, "text": text}]
            - Generator and datasets
            This is a common pattern in other multimodal pipelines, so we support it here as well.
            """
            return super().__call__(images, **kwargs)

        if isinstance(text, (list, tuple, KeyDataset)) and isinstance(text[0], (list, tuple, dict)):
            # We have one or more prompts in list-of-dicts format, so this is chat mode
            if isinstance(text[0], dict):
                return super().__call__(Chat(text, images), **kwargs)
            else:
                if images is None:
                    images = [None] * len(text)
                chats = [Chat(chat, image) for chat, image in zip(text, images)]  # 🐈 🐈 🐈
                return super().__call__(chats, **kwargs)

        # encourage the user to use the chat format if supported
        if getattr(self.processor, "chat_template", None) is not None:
            logger.warning_once(
                "The input data was not formatted as a chat with dicts containing 'role' and 'content' keys, even though this model supports chat. "
                "Consider using the chat format for better results. For more information, see https://huggingface.co/docs/transformers/en/chat_templating"
            )

        # support text only generation
        if images is None:
            return super().__call__(text, **kwargs)
        if text is None:
            raise ValueError("You must provide text for this pipeline.")

        return super().__call__({"images": images, "text": text}, **kwargs)

    def preprocess(self, inputs=None, timeout=None, continue_final_message=None, processing_kwargs=None):
        # In case we only have text inputs
        if isinstance(inputs, (list, tuple, str)):
            images = None
            text = inputs
            inputs_text = inputs
        else:
            if isinstance(inputs, Chat):
                # If the user passes a chat that ends in an assistant message, we treat it as a prefill by default
                # because very few models support multiple separate, consecutive assistant messages
                if continue_final_message is None:
                    continue_final_message = inputs.messages[-1]["role"] == "assistant"
                text = self.processor.apply_chat_template(
                    inputs.messages,
                    add_generation_prompt=not continue_final_message,
                    continue_final_message=continue_final_message,
                    return_tensors=self.framework,
                )
                inputs_text = inputs
                images = inputs.images
            else:
                text = inputs["text"]
                inputs_text = inputs["text"]
                images = inputs["images"]

            images = load_images(images)

        # if batched text inputs, we set padding to True unless specified otherwise
        if isinstance(text, (list, tuple)) and len(text) > 1:
            processing_kwargs.setdefault("padding", True)
        model_inputs = self.processor(
            images=images, text=text, return_tensors=self.framework, legacy=False, **processing_kwargs
        ).to(dtype=self.torch_dtype)

        model_inputs["text"] = inputs_text

        return model_inputs

    def _forward(self, model_inputs, generate_kwargs=None):
        generate_kwargs = {} if generate_kwargs is None else generate_kwargs
        prompt_text = model_inputs.pop("text")
        input_ids = (
            model_inputs["input_ids"] if "input_ids" in model_inputs else model_inputs["decoder_input_ids"]
        )  # for decoder-only models
        generated_sequence = self.model.generate(**model_inputs, **generate_kwargs)

        return {"generated_sequence": generated_sequence, "prompt_text": prompt_text, "input_ids": input_ids}

    def postprocess(self, model_outputs, return_type=ReturnType.FULL_TEXT, continue_final_message=None):
        input_texts = model_outputs["prompt_text"]
        input_texts = [input_texts] if isinstance(input_texts, (str, Chat)) else input_texts
        generated_sequence = model_outputs["generated_sequence"]
        input_ids = model_outputs["input_ids"]
        if return_type == ReturnType.TENSORS:
            return [
                {"input_text": input_texts[i], "generated_token_ids": generated_sequence[i]}
                for i in range(len(input_texts))
            ]

        # Decode inputs and outputs the same way to remove input text from generated text if present
        generated_texts = self.processor.post_process_image_text_to_text(generated_sequence)
        decoded_inputs = self.processor.post_process_image_text_to_text(input_ids)

        # Force consistent behavior for including the input text in the output
        if return_type in {ReturnType.NEW_TEXT, ReturnType.FULL_TEXT}:
            # Remove the input text from the generated text if the generated text starts with the input text
            # (accounting for the possibility of a space between the input and generated text)
            new_generated_texts = []
            for text_generated, decoded_input in zip(generated_texts, decoded_inputs):
                # There can be added characters before the input text, so we need to find the beginning of the input text in the generated text
                index_input_text = text_generated.find(decoded_input)
                # Limit the search to 2 residual characters, like spaces or new lines, to avoid removing a large part of the answer
                if 0 <= index_input_text <= 2:
                    # If the input text is found, we remove it
                    new_generated_texts.append(text_generated[index_input_text + len(decoded_input) :])
                else:
                    new_generated_texts.append(text_generated)
            generated_texts = new_generated_texts
        if return_type == ReturnType.FULL_TEXT:
            full_texts = []
            for prompt_text, generated_text in zip(input_texts, generated_texts):
                if isinstance(prompt_text, str):
                    generated_text = prompt_text + generated_text
                elif isinstance(prompt_text, Chat):
                    if continue_final_message is None:
                        # If the user passes a chat ending in an assistant message, we treat it as a prefill by
                        # default because very few models support multiple separate, consecutive assistant messages
                        continue_final_message = prompt_text.messages[-1]["role"] == "assistant"
                    if continue_final_message:
                        # With assistant prefill, concat onto the end of the last message
                        new_text = dict(prompt_text.messages[-1]["content"][-1].items())
                        new_text["text"] += generated_text
                        generated_text = list(prompt_text.messages)[:-1] + [
                            {
                                "role": prompt_text.messages[-1]["role"],
                                "content": prompt_text.messages[-1]["content"][:-1] + [new_text],
                            }
                        ]
                    else:
                        # When we're not starting from a prefill, the output is a new assistant message
                        generated_text = list(prompt_text.messages) + [
                            {"role": "assistant", "content": generated_text}
                        ]
                full_texts.append(generated_text)
            generated_texts = full_texts

        records = [
            {
                "input_text": input_text.messages if isinstance(input_text, Chat) else input_text,
                "generated_text": generated_text,
            }
            for input_text, generated_text in zip(input_texts, generated_texts)
        ]

        return records
