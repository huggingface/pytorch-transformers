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

from typing import List

from ...processing_utils import ProcessorMixin
from ...utils import is_torch_available


if is_torch_available():
    import torch


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
        max_seq_len (`int`, *optional*, defaults to 77)):
            Sequence length for input text list.
        task_seq_len (`int`, *optional*, defaults to 77):
            Sequence length for input task token.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "OneFormerImageProcessor"
    tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")

    def __init__(
        self, image_processor=None, tokenizer=None, max_seq_length: int = 77, task_seq_length: int = 77, **kwargs
    ):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        self.max_seq_length = max_seq_length
        self.task_seq_length = task_seq_length

        super().__init__(image_processor, tokenizer)

    def _preprocess_text(self, text_list=None, max_length=77):
        if text_list is None:
            raise ValueError("tokens cannot be None.")

        tokens = self.tokenizer(text_list, padding="max_length", max_length=max_length, truncation=True)

        attention_masks, input_ids = tokens["attention_mask"], tokens["input_ids"]

        token_inputs = []
        for attn_mask, input_id in zip(attention_masks, input_ids):
            token = torch.tensor(attn_mask) * torch.tensor(input_id)
            token_inputs.append(token.unsqueeze(0))

        token_inputs = torch.cat(token_inputs, dim=0)
        return token_inputs

    def __call__(self, images=None, task_inputs=None, segmentation_maps=None, **kwargs):
        """
        Main method to prepare for the model one or several task input(s) and image(s). This method forwards the
        `task_inputs` and `kwargs` arguments to CLIPTokenizer's [`~CLIPTokenizer.__call__`] if `task_inputs` is not
        `None` to encode. To prepare the image(s), this method forwards the `images` and `kwargs` arguments to
        OneFormerImageProcessor's [`~OneFormerImageProcessor.__call__`] if `images` is not `None`. Please refer to the
        doctsring of the above two methods for more information.

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

        if task_inputs is None:
            raise ValueError("You have to specify the task_input. Found None.")
        elif images is None:
            raise ValueError("You have to specify the image. Found None.")

        if not all(task in ["semantic", "instance", "panoptic"] for task in task_inputs):
            raise ValueError("task_inputs must be semantic, instance, or panoptic.")

        encoded_inputs = self.image_processor(images, task_inputs, segmentation_maps, **kwargs)

        if isinstance(task_inputs, str):
            task_inputs = [task_inputs]

        if isinstance(task_inputs, List) and all(isinstance(task_input, str) for task_input in task_inputs):
            task_token_inputs = []
            for task in task_inputs:
                task_input = f"the task is {task}"
                task_token_inputs.append(task_input)
            encoded_inputs["task_inputs"] = self._preprocess_text(task_token_inputs, max_length=self.task_seq_length)
        else:
            raise TypeError("Task Inputs should be a string or a list of strings.")

        if hasattr(encoded_inputs, "text_inputs"):
            texts_list = encoded_inputs.text_inputs

            text_inputs = []
            for texts in texts_list:
                text_input_list = self._preprocess_text(texts, max_length=self.max_seq_length)
                text_inputs.append(text_input_list.unsqueeze(0))

            encoded_inputs["text_inputs"] = torch.cat(text_inputs, dim=0)

        return encoded_inputs

    def encode_inputs(self, images=None, task_inputs=None, segmentation_maps=None, **kwargs):
        """
        This method forwards all its arguments to [`OneFormerImageProcessor.encode_inputs`] and then tokenizes the
        task_inputs. Please refer to the docstring of this method for more information.
        """

        if task_inputs is None:
            raise ValueError("You have to specify the task_input. Found None.")
        elif images is None:
            raise ValueError("You have to specify the image. Found None.")

        if not all(task in ["semantic", "instance", "panoptic"] for task in task_inputs):
            raise ValueError("task_inputs must be semantic, instance, or panoptic.")

        encoded_inputs = self.image_processor.encode_inputs(images, task_inputs, segmentation_maps, **kwargs)

        if isinstance(task_inputs, str):
            task_inputs = [task_inputs]

        if isinstance(task_inputs, List) and all(isinstance(task_input, str) for task_input in task_inputs):
            task_token_inputs = []
            for task in task_inputs:
                task_input = f"the task is {task}"
                task_token_inputs.append(task_input)
            encoded_inputs["task_inputs"] = self._preprocess_text(task_token_inputs, max_length=self.task_seq_length)
        else:
            raise TypeError("Task Inputs should be a string or a list of strings.")

        if hasattr(encoded_inputs, "text_inputs"):
            texts_list = encoded_inputs.text_inputs

            text_inputs = []
            for texts in texts_list:
                text_input_list = self._preprocess_text(texts, max_length=self.max_seq_length)
                text_inputs.append(text_input_list.unsqueeze(0))

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
