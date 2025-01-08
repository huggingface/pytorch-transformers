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

from typing import List, Union

from ..utils import (
    add_end_docstrings,
    is_tf_available,
    is_torch_available,
    is_vision_available,
    logging,
    requires_backends,
)
from .base import Pipeline, build_pipeline_init_args


if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image

if is_tf_available():
    from ..models.auto.modeling_tf_auto import TF_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES

if is_torch_available():
    import torch

    from ..models.auto.modeling_auto import MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES

logger = logging.get_logger(__name__)


@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True, has_image_processor=True))
class ImageToTextPipeline(Pipeline):
    """
    Image To Text pipeline using a `AutoModelForVision2Seq`. This pipeline predicts a caption for a given image.

    Example:

    ```python
    >>> from transformers import pipeline

    >>> captioner = pipeline(model="ydshieh/vit-gpt2-coco-en")
    >>> captioner("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png")
    [{'generated_text': 'two birds are standing next to each other '}]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This image to text pipeline can currently be loaded from pipeline() using the following task identifier:
    "image-to-text".

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?pipeline_tag=image-to-text).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        requires_backends(self, "vision")
        self.check_model_type(
            TF_MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES if self.framework == "tf" else MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES
        )

    def _sanitize_parameters(self, max_new_tokens=None, generate_kwargs=None, prompt=None, timeout=None):
        forward_params = {}
        preprocess_params = {}

        if prompt is not None:
            preprocess_params["prompt"] = prompt
        if timeout is not None:
            preprocess_params["timeout"] = timeout

        if max_new_tokens is not None:
            forward_params["max_new_tokens"] = max_new_tokens
        if generate_kwargs is not None:
            if max_new_tokens is not None and "max_new_tokens" in generate_kwargs:
                raise ValueError(
                    "`max_new_tokens` is defined both as an argument and inside `generate_kwargs` argument, please use"
                    " only 1 version"
                )
            forward_params.update(generate_kwargs)

        if self.assistant_model is not None:
            forward_params["assistant_model"] = self.assistant_model
        if self.assistant_tokenizer is not None:
            forward_params["tokenizer"] = self.tokenizer
            forward_params["assistant_tokenizer"] = self.assistant_tokenizer

        return preprocess_params, forward_params, {}

    def __call__(self, inputs: Union[str, List[str], "Image.Image", List["Image.Image"]] = None, **kwargs):
        """
        Assign labels to the image(s) passed as inputs.

        Args:
            inputs (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a HTTP(s) link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images.

            max_new_tokens (`int`, *optional*):
                The amount of maximum tokens to generate. By default it will use `generate` default.

            generate_kwargs (`Dict`, *optional*):
                Pass it to send all of these arguments directly to `generate` allowing full control of this function.

            timeout (`float`, *optional*, defaults to None):
                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
                the call may block forever.

        Return:
            A list or a list of list of `dict`: Each result comes as a dictionary with the following key:

            - **generated_text** (`str`) -- The generated text.
        """
        # After deprecation of this is completed, remove the default `None` value for `images`
        if "images" in kwargs:
            inputs = kwargs.pop("images")
        if inputs is None:
            raise ValueError("Cannot call the image-to-text pipeline without an inputs argument!")
        return super().__call__(inputs, **kwargs)

    def preprocess(self, image, prompt=None, timeout=None):
        image = load_image(image, timeout=timeout)

        if prompt is not None:
            logger.warning_once(
                "Passing `prompt` to the `image-to-text` pipeline is deprecated and will be removed in version 4.48"
                " of 🤗 Transformers. Use the `image-text-to-text` pipeline instead",
            )
            if not isinstance(prompt, str):
                raise ValueError(
                    f"Received an invalid text input, got - {type(prompt)} - but expected a single string. "
                    "Note also that one single text can be provided for conditional image to text generation."
                )

            model_type = self.model.config.model_type

            if model_type == "git":
                model_inputs = self.image_processor(images=image, return_tensors=self.framework)
                if self.framework == "pt":
                    model_inputs = model_inputs.to(self.torch_dtype)
                input_ids = self.tokenizer(text=prompt, add_special_tokens=False).input_ids
                input_ids = [self.tokenizer.cls_token_id] + input_ids
                input_ids = torch.tensor(input_ids).unsqueeze(0)
                model_inputs.update({"input_ids": input_ids})

            elif model_type == "pix2struct":
                model_inputs = self.image_processor(images=image, header_text=prompt, return_tensors=self.framework)
                if self.framework == "pt":
                    model_inputs = model_inputs.to(self.torch_dtype)

            elif model_type != "vision-encoder-decoder":
                # vision-encoder-decoder does not support conditional generation
                model_inputs = self.image_processor(images=image, return_tensors=self.framework)
                if self.framework == "pt":
                    model_inputs = model_inputs.to(self.torch_dtype)
                text_inputs = self.tokenizer(prompt, return_tensors=self.framework)
                model_inputs.update(text_inputs)

            else:
                raise ValueError(f"Model type {model_type} does not support conditional text generation")

        else:
            model_inputs = self.image_processor(images=image, return_tensors=self.framework)
            if self.framework == "pt":
                model_inputs = model_inputs.to(self.torch_dtype)

        if self.model.config.model_type == "git" and prompt is None:
            model_inputs["input_ids"] = None

        return model_inputs

    def _forward(self, model_inputs, **generate_kwargs):
        # Git model sets `model_inputs["input_ids"] = None` in `preprocess` (when `prompt=None`). In batch model, the
        # pipeline will group them into a list of `None`, which fail `_forward`. Avoid this by checking it first.
        if (
            "input_ids" in model_inputs
            and isinstance(model_inputs["input_ids"], list)
            and all(x is None for x in model_inputs["input_ids"])
        ):
            model_inputs["input_ids"] = None

        # User-defined `generation_config` passed to the pipeline call take precedence
        if "generation_config" not in generate_kwargs:
            generate_kwargs["generation_config"] = self.generation_config

        # FIXME: We need to pop here due to a difference in how `generation.py` and `generation.tf_utils.py`
        #  parse inputs. In the Tensorflow version, `generate` raises an error if we don't use `input_ids` whereas
        #  the PyTorch version matches it with `self.model.main_input_name` or `self.model.encoder.main_input_name`
        #  in the `_prepare_model_inputs` method.
        inputs = model_inputs.pop(self.model.main_input_name)
        model_outputs = self.model.generate(inputs, **model_inputs, **generate_kwargs)
        return model_outputs

    def postprocess(self, model_outputs):
        records = []
        for output_ids in model_outputs:
            record = {
                "generated_text": self.tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                )
            }
            records.append(record)
        return records
