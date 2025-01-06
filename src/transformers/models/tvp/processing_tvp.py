# coding=utf-8
# Copyright 2023 The Intel AIA Team Authors, and HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License=, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing=, software
# distributed under the License is distributed on an "AS IS" BASIS=,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND=, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for TVP.
"""

from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import BatchEncoding


class TvpProcessor(ProcessorMixin):
    r"""
    Constructs an TVP processor which wraps a TVP image processor and a Bert tokenizer into a single processor.

    [`TvpProcessor`] offers all the functionalities of [`TvpImageProcessor`] and [`BertTokenizerFast`]. See the
    [`~TvpProcessor.__call__`] and [`~TvpProcessor.decode`] for more information.

    Args:
        image_processor ([`TvpImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`BertTokenizerFast`], *optional*):
            The tokenizer is a required input.
    """

    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "TvpImageProcessor"
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, **kwargs):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")

        super().__init__(image_processor, tokenizer)

    def __call__(self, text=None, videos=None, return_tensors=None, **kwargs):
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to BertTokenizerFast's [`~BertTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `videos` and `kwargs` arguments to
        TvpImageProcessor's [`~TvpImageProcessor.__call__`] if `videos` is not `None`. Please refer to the doctsring of
        the above two methods for more information.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`, `List[List[PIL.Image.Image]]`, `List[List[np.ndarrray]]`,:
                `List[List[torch.Tensor]]`): The video or batch of videos to be prepared. Each video should be a list
                of frames, which can be either PIL images or NumPy arrays. In case of NumPy arrays/PyTorch tensors,
                each frame should be of shape (H, W, C), where H and W are frame height and width, and C is a number of
                channels.

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
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `videos` is not `None`.
        """

        max_text_length = kwargs.pop("max_text_length", None)

        if text is None and videos is None:
            raise ValueError("You have to specify either text or videos. Both cannot be none.")

        encoding = {}
        if text is not None:
            textual_input = self.tokenizer.batch_encode_plus(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_text_length,
                pad_to_max_length=True,
                return_tensors=return_tensors,
                return_token_type_ids=False,
                **kwargs,
            )
            encoding.update(textual_input)

        if videos is not None:
            image_features = self.image_processor(videos, return_tensors=return_tensors, **kwargs)
            encoding.update(image_features)

        return BatchEncoding(data=encoding, tensor_type=return_tensors)

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    def post_process_video_grounding(self, logits, video_durations):
        """
        Compute the time of the video.

        Args:
            logits (`torch.Tensor`):
                The logits output of TvpForVideoGrounding.
            video_durations (`float`):
                The video's duration.

        Returns:
            start (`float`):
                The start time of the video.
            end (`float`):
                The end time of the video.
        """
        start, end = (
            round(logits.tolist()[0][0] * video_durations, 1),
            round(logits.tolist()[0][1] * video_durations, 1),
        )

        return start, end

    @property
    # Copied from transformers.models.blip.processing_blip.BlipProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["TvpProcessor"]
