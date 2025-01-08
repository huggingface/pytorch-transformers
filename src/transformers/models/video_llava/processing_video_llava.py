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
Processor class for VideoLlava.
"""

from typing import List, Optional, Union

from ...feature_extraction_utils import BatchFeature
from ...image_utils import ImageInput, get_image_size, to_numpy_array
from ...processing_utils import ProcessorMixin
from ...tokenization_utils_base import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType, logging


logger = logging.get_logger(__name__)


class VideoLlavaProcessor(ProcessorMixin):
    r"""
    Constructs a VideoLlava processor which wraps a VideoLlava image processor and a Llava tokenizer into a single processor.

    [`VideoLlavaProcessor`] offers all the functionalities of [`VideoLlavaImageProcessor`] and [`LlamaTokenizerFast`]. See the
    [`~VideoLlavaProcessor.__call__`] and [`~VideoLlavaProcessor.decode`] for more information.

    Args:
        image_processor ([`VideoLlavaImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`LlamaTokenizerFast`], *optional*):
            The tokenizer is a required input.
        patch_size (`int`, *optional*, defaults to 14):
            Patch size from the vision tower.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Shoudl be same as in model's config
        image_token (`str`, *optional*, defaults to `"<image>"`):
            Special token used to denote image location.
        video_token (`str`, *optional*, defaults to `"<video>"`):
            Special token used to denote video location.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
        num_additional_image_tokens (`int`, *optional*, defaults to 1):
            Number of additional tokens added to the image embeddings, such as CLS (+1). If the backbone has no CLS or other
            extra tokens appended, no need to set this arg.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = [
        "chat_template",
        "patch_size",
        "vision_feature_select_strategy",
        "image_token",
        "video_token",
        "num_additional_image_tokens",
    ]
    image_processor_class = "VideoLlavaImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        patch_size=14,
        vision_feature_select_strategy="default",
        image_token="<image>",  # set the default and let users change if they have peculiar special tokens in rare cases
        video_token="<video>",
        chat_template=None,
        num_additional_image_tokens=1,
        **kwargs,
    ):
        self.patch_size = patch_size
        self.num_additional_image_tokens = num_additional_image_tokens
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.image_token = tokenizer.image_token if hasattr(tokenizer, "image_token") else image_token
        self.video_token = tokenizer.video_token if hasattr(tokenizer, "video_token") else video_token
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        images: ImageInput = None,
        videos: ImageInput = None,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to LlamaTokenizerFast's [`~LlamaTokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the image(s), this method forwards the `images` and `kwrags` arguments to
        VideoLlavaImageProcessor's [`~VideoLlavaImageProcessor.__call__`] if `images` is not `None`. Please refer to the doctsring
        of the above two methods for more information.

        Args:
            text (`TextInput`, `PreTokenizedInput`, `List[TextInput]`, `List[PreTokenizedInput]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. In case of a NumPy array/PyTorch tensor, each image should be of shape (C, H, W), where C is a
                number of channels, H and W are image height and width.
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
                Video frames to preprocess. Expects a single or batch of video frames in NumPy array or PyTorch
                tensor. Each video should be of shape (T, C, H, W), where T is number of frames, C is
                number of channels, H and W are image height and width.
            padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `False`):
                Select a strategy to pad the returned sequences (according to the model's padding side and padding
                index) among:
                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            max_length (`int`, *optional*):
                Maximum length of the returned list and optionally padding length (see above).
            truncation (`bool`, *optional*):
                Activates truncation to cut input sequences longer than `max_length` to `max_length`.
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
        data = {}
        if images is not None or videos is not None:
            encoded_images = self.image_processor(images=images, videos=videos, return_tensors=return_tensors)
            data.update(encoded_images)

        if isinstance(text, str):
            text = [text]
        elif not isinstance(text, list) and not isinstance(text[0], str):
            raise ValueError("Invalid input text. Please provide a string, or a list of strings")

        prompt_strings = text

        if encoded_images is not None:
            if "pixel_values_images" in encoded_images.keys():
                height, width = get_image_size(to_numpy_array(encoded_images.get("pixel_values_images")[0]))
                num_frames = 1

            if "pixel_values_videos" in encoded_images.keys():
                one_video = to_numpy_array(encoded_images.get("pixel_values_videos")[0])
                height, width = get_image_size(one_video[0])
                num_frames = one_video.shape[0]  # frame dim is always after batch dim

            num_image_tokens = (height // self.patch_size) * (
                width // self.patch_size
            ) + self.num_additional_image_tokens
            num_video_tokens = num_image_tokens * num_frames

            num_image_tokens = (height // self.patch_size) * (
                width // self.patch_size
            ) + self.num_additional_image_tokens
            num_video_tokens = num_image_tokens * num_frames
            if self.vision_feature_select_strategy == "default":
                num_image_tokens -= 1

            prompt_strings = []
            for sample in text:
                sample = sample.replace(self.image_token, self.image_token * num_image_tokens)
                sample = sample.replace(self.video_token, self.video_token * num_video_tokens)
                prompt_strings.append(sample)

        text_inputs = self.tokenizer(
            prompt_strings,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
        data.update(text_inputs)

        return BatchFeature(data=data)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Llama
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Llama
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to LlamaTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))


__all__ = ["VideoLlavaProcessor"]
