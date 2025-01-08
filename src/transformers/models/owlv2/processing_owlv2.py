# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
Image/Text processor class for OWLv2
"""

from typing import List

import numpy as np

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding
from ...utils import is_flax_available, is_tf_available, is_torch_available


class Owlv2Processor(ProcessorMixin):
    r"""
    Constructs an Owlv2 processor which wraps [`Owlv2ImageProcessor`] and [`CLIPTokenizer`]/[`CLIPTokenizerFast`] into
    a single processor that interits both the image processor and tokenizer functionalities. See the
    [`~OwlViTProcessor.__call__`] and [`~OwlViTProcessor.decode`] for more information.

    Args:
        image_processor ([`Owlv2ImageProcessor`]):
            The image processor is a required input.
        tokenizer ([`CLIPTokenizer`, `CLIPTokenizerFast`]):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "Owlv2ImageProcessor"
    tokenizer_class = ("CLIPTokenizer", "CLIPTokenizerFast")

    def __init__(self, image_processor, tokenizer, **kwargs):
        super().__init__(image_processor, tokenizer)

    # Copied from transformers.models.owlvit.processing_owlvit.OwlViTProcessor.__call__ with OWLViT->OWLv2
    def __call__(self, text=None, images=None, query_images=None, padding="max_length", return_tensors="np", **kwargs):
        """
        Main method to prepare for the model one or several text(s) and image(s). This method forwards the `text` and
        `kwargs` arguments to CLIPTokenizerFast's [`~CLIPTokenizerFast.__call__`] if `text` is not `None` to encode:
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        CLIPImageProcessor's [`~CLIPImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`,
            `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            query_images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The query image to be prepared, one query image is expected per target image to be queried. Each image
                can be a PIL image, NumPy array or PyTorch tensor. In case of a NumPy array/PyTorch tensor, each image
                should be of shape (C, H, W), where C is a number of channels, H and W are image height and width.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.
        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:
            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
        """

        if text is None and query_images is None and images is None:
            raise ValueError(
                "You have to specify at least one text or query image or image. All three cannot be none."
            )

        if text is not None:
            if isinstance(text, str) or (isinstance(text, List) and not isinstance(text[0], List)):
                encodings = [self.tokenizer(text, padding=padding, return_tensors=return_tensors, **kwargs)]

            elif isinstance(text, List) and isinstance(text[0], List):
                encodings = []

                # Maximum number of queries across batch
                max_num_queries = max([len(t) for t in text])

                # Pad all batch samples to max number of text queries
                for t in text:
                    if len(t) != max_num_queries:
                        t = t + [" "] * (max_num_queries - len(t))

                    encoding = self.tokenizer(t, padding=padding, return_tensors=return_tensors, **kwargs)
                    encodings.append(encoding)
            else:
                raise TypeError("Input text should be a string, a list of strings or a nested list of strings")

            if return_tensors == "np":
                input_ids = np.concatenate([encoding["input_ids"] for encoding in encodings], axis=0)
                attention_mask = np.concatenate([encoding["attention_mask"] for encoding in encodings], axis=0)

            elif return_tensors == "jax" and is_flax_available():
                import jax.numpy as jnp

                input_ids = jnp.concatenate([encoding["input_ids"] for encoding in encodings], axis=0)
                attention_mask = jnp.concatenate([encoding["attention_mask"] for encoding in encodings], axis=0)

            elif return_tensors == "pt" and is_torch_available():
                import torch

                input_ids = torch.cat([encoding["input_ids"] for encoding in encodings], dim=0)
                attention_mask = torch.cat([encoding["attention_mask"] for encoding in encodings], dim=0)

            elif return_tensors == "tf" and is_tf_available():
                import tensorflow as tf

                input_ids = tf.stack([encoding["input_ids"] for encoding in encodings], axis=0)
                attention_mask = tf.stack([encoding["attention_mask"] for encoding in encodings], axis=0)

            else:
                raise ValueError("Target return tensor type could not be returned")

            encoding = BatchEncoding()
            encoding["input_ids"] = input_ids
            encoding["attention_mask"] = attention_mask

        if query_images is not None:
            encoding = BatchEncoding()
            query_pixel_values = self.image_processor(
                query_images, return_tensors=return_tensors, **kwargs
            ).pixel_values
            encoding["query_pixel_values"] = query_pixel_values

        if images is not None:
            image_features = self.image_processor(images, return_tensors=return_tensors, **kwargs)

        if text is not None and images is not None:
            encoding["pixel_values"] = image_features.pixel_values
            return encoding
        elif query_images is not None and images is not None:
            encoding["pixel_values"] = image_features.pixel_values
            return encoding
        elif text is not None or query_images is not None:
            return encoding
        else:
            return BatchEncoding(data=dict(**image_features), tensor_type=return_tensors)

    # Copied from transformers.models.owlvit.processing_owlvit.OwlViTProcessor.post_process_object_detection with OWLViT->OWLv2
    def post_process_object_detection(self, *args, **kwargs):
        """
        This method forwards all its arguments to [`OwlViTImageProcessor.post_process_object_detection`]. Please refer
        to the docstring of this method for more information.
        """
        return self.image_processor.post_process_object_detection(*args, **kwargs)

    # Copied from transformers.models.owlvit.processing_owlvit.OwlViTProcessor.post_process_image_guided_detection with OWLViT->OWLv2
    def post_process_image_guided_detection(self, *args, **kwargs):
        """
        This method forwards all its arguments to [`OwlViTImageProcessor.post_process_one_shot_object_detection`].
        Please refer to the docstring of this method for more information.
        """
        return self.image_processor.post_process_image_guided_detection(*args, **kwargs)

    # Copied from transformers.models.owlvit.processing_owlvit.OwlViTProcessor.batch_decode
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.owlvit.processing_owlvit.OwlViTProcessor.decode
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)


__all__ = ["Owlv2Processor"]
