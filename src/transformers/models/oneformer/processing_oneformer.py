# coding=utf-8
# Copyright 2022 SHI Labs and The HuggingFace Inc. team.
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
Image/Text processor class for OneFormer
"""

from typing import Dict, List, Optional, Union

from ...image_processing_utils import BatchFeature
from ...image_utils import ImageInput
from ...processing_utils import ImagesKwargs, ProcessingKwargs, ProcessorMixin, Unpack
from ...tokenization_utils_base import PreTokenizedInput, TextInput
from ...utils import is_torch_available


if is_torch_available():
    import torch


class OneFormerImagesKwargs(ImagesKwargs):
    segmentation_maps: Optional[ImageInput]
    task_inputs: Optional[Union[TextInput, PreTokenizedInput]]
    instance_id_to_semantic_id: Optional[Dict[int, int]]
    pad_and_return_pixel_mask: Optional[bool]
    ignore_index: Optional[int]
    do_reduce_labels: bool
    repo_path: Optional[str]
    class_info_file: Optional[str]
    num_text: Optional[int]
    num_labels: Optional[int]


class OneFormerProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: OneFormerImagesKwargs
    _defaults = {
        "text_kwargs": {
            "padding": "max_length",
            "truncation": True,
        },
    }


class OneFormerProcessor(ProcessorMixin):
    r"""
    Constructs an OneFormer processor which wraps [`OneFormerImageProcessor`] and
    [`CLIPTokenizer`]/[`CLIPTokenizerFast`] into a single processor that inherits both the image processor and
    tokenizer functionalities.

    Args:
        image_processor ([`OneFormerImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`CLIPTokenizer`, `CLIPTokenizerFast`]):
            The tokenizer is a required input.
        max_seq_length (`int`, *optional*, defaults to 77):
            Sequence length for input text list.
        task_seq_length (`int`, *optional*, defaults to 77):
            Sequence length for input task token.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "OneFormerImageProcessor"
    tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")
    optional_call_args = ["task_inputs", "segmentation_maps"]

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        max_seq_length: int = 77,
        task_seq_length: int = 77,
        **kwargs,
    ):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        self.max_seq_length = max_seq_length
        self.task_seq_length = task_seq_length

        super().__init__(image_processor, tokenizer)

    def _preprocess_text(
        self,
        text_list: PreTokenizedInput,
        max_length: int = 77,
        text_kwargs: Optional[dict] = None,
    ):
        if text_kwargs is None:
            text_kwargs = {}
        if "max_length" not in text_kwargs:
            text_kwargs["max_length"] = max_length
        tokens = self.tokenizer(text_list, **text_kwargs)

        attention_masks, input_ids = tokens["attention_mask"], tokens["input_ids"]

        token_inputs = []
        for attn_mask, input_id in zip(attention_masks, input_ids):
            token = torch.tensor(attn_mask) * torch.tensor(input_id)
            token_inputs.append(token.unsqueeze(0))

        token_inputs = torch.cat(token_inputs, dim=0)
        return token_inputs

    @staticmethod
    def _validate_input_types(
        images: Optional[ImageInput] = None,
        task_inputs: Optional[PreTokenizedInput] = None,
    ) -> None:
        if task_inputs is None:
            raise ValueError("You have to specify the task_inputs. Found None.")
        elif images is None:
            raise ValueError("You have to specify the images. Found None.")

        if not isinstance(task_inputs, List) or not task_inputs:
            raise TypeError("task_inputs should be a string or a list of strings.")

        if not all(task in ["semantic", "instance", "panoptic"] for task in task_inputs):
            raise ValueError("task_inputs must be semantic, instance, or panoptic.")

    def __call__(
        self,
        images: Optional[ImageInput] = None,
        # The following is to capture `task_inputs and `segmentation_maps` arguments
        # that may be passed as a positional argument.
        # See transformers.processing_utils.ProcessorMixin.prepare_and_validate_optional_call_args for more details,
        # or this conversation for more context:
        # https://github.com/huggingface/transformers/pull/32544#discussion_r1720208116
        # This behavior is only needed for backward compatibility and will be removed in future versions.
        *args,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[OneFormerProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several task input(s) and image(s). This method forwards the
        `task_inputs` and `kwargs` arguments to CLIPTokenizer's [`~CLIPTokenizer.__call__`] if `task_inputs` is not
        `None` to encode. To prepare the image(s), this method forwards the `images` and `kwargs` arguments to
        OneFormerImageProcessor's [`~OneFormerImageProcessor.__call__`] if `images` is not `None`. Please refer to the
        docstring of the above two methods for more information.

        Args:
            task_inputs (`str`, `List[str]`):
                The sequence or batch of task_inputs sequences to be encoded. Each sequence can be a string or a list
                of strings of the template "the task is {task}".
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`,
            `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            segmentation_maps (`ImageInput`, *optional*):
                The corresponding semantic segmentation maps with the pixel-wise annotations.

             (`bool`, *optional*, defaults to `True`):
                Whether or not to pad images up to the largest image in a batch and create a pixel mask.

                If left to the default, will return a pixel mask that is:

                - 1 for pixels that are real (i.e. **not masked**),
                - 0 for pixels that are padding (i.e. **masked**).
        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:
            - **task_inputs** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """

        output_kwargs = self._merge_kwargs(
            OneFormerProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
            **self.prepare_and_validate_optional_call_args(*args),
        )
        task_inputs = output_kwargs["images_kwargs"].pop("task_inputs", None)
        segmentation_maps = output_kwargs["images_kwargs"].pop("segmentation_maps", None)
        if isinstance(task_inputs, str):
            task_inputs = [task_inputs]
        self._validate_input_types(images, task_inputs)

        encoded_inputs = self.image_processor(images, task_inputs, segmentation_maps, **output_kwargs["images_kwargs"])

        task_token_inputs = []
        for task in task_inputs:
            task_input = f"the task is {task}"
            task_token_inputs.append(task_input)
        encoded_inputs["task_inputs"] = self._preprocess_text(
            task_token_inputs,
            max_length=self.task_seq_length,
            text_kwargs=output_kwargs["text_kwargs"],
        )

        if hasattr(encoded_inputs, "text_inputs"):
            text_inputs = []
            for texts in encoded_inputs.text_inputs:
                text_inputs.append(
                    self._preprocess_text(
                        texts,
                        max_length=self.max_seq_length,
                        text_kwargs=output_kwargs["text_kwargs"],
                    ).unsqueeze(0)
                )
            encoded_inputs["text_inputs"] = torch.cat(text_inputs, dim=0)

        return encoded_inputs

    def encode_inputs(
        self,
        images=None,
        task_inputs=None,
        segmentation_maps=None,
        **kwargs,
    ):
        """
        This method forwards all its arguments to [`OneFormerImageProcessor.encode_inputs`] and then tokenizes the
        task_inputs. Please refer to the docstring of this method for more information.
        """

        if isinstance(task_inputs, str):
            task_inputs = [task_inputs]
        self._validate_input_types(images, task_inputs)

        encoded_inputs = self.image_processor.encode_inputs(images, task_inputs, segmentation_maps, **kwargs)

        task_token_inputs = []
        for task in task_inputs:
            task_input = f"the task is {task}"
            task_token_inputs.append(task_input)
        text_kwargs = {
            "padding": "max_length",
            "truncation": True,
        }
        encoded_inputs["task_inputs"] = self._preprocess_text(
            task_token_inputs,
            max_length=self.task_seq_length,
            text_kwargs=text_kwargs,
        )

        if hasattr(encoded_inputs, "text_inputs"):
            text_inputs = []
            for texts in encoded_inputs.text_inputs:
                text_inputs.append(
                    self._preprocess_text(
                        texts,
                        max_length=self.max_seq_length,
                        text_kwargs=text_kwargs,
                    ).unsqueeze(0)
                )
            encoded_inputs["text_inputs"] = torch.cat(text_inputs, dim=0)

        return encoded_inputs

    def post_process_semantic_segmentation(self, *args, **kwargs):
        """
        This method forwards all its arguments to [`OneFormerImageProcessor.post_process_semantic_segmentation`].
        Please refer to the docstring of this method for more information.
        """
        return self.image_processor.post_process_semantic_segmentation(*args, **kwargs)

    def post_process_instance_segmentation(self, *args, **kwargs):
        """
        This method forwards all its arguments to [`OneFormerImageProcessor.post_process_instance_segmentation`].
        Please refer to the docstring of this method for more information.
        """
        return self.image_processor.post_process_instance_segmentation(*args, **kwargs)

    def post_process_panoptic_segmentation(self, *args, **kwargs):
        """
        This method forwards all its arguments to [`OneFormerImageProcessor.post_process_panoptic_segmentation`].
        Please refer to the docstring of this method for more information.
        """
        return self.image_processor.post_process_panoptic_segmentation(*args, **kwargs)


__all__ = ["OneFormerProcessor"]
