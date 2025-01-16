# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
Processing saving/loading class for common processors.
"""

import copy
import inspect
import json
import os
import sys
import typing
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import numpy as np
import typing_extensions

from .dynamic_module_utils import custom_object_save
from .image_utils import ChannelDimension, is_valid_image, is_vision_available, load_image, load_video


if is_vision_available():
    from .image_utils import PILImageResampling

from .tokenization_utils_base import (
    PaddingStrategy,
    PreTokenizedInput,
    PreTrainedTokenizerBase,
    TextInput,
    TruncationStrategy,
)
from .utils import (
    PROCESSOR_NAME,
    PushToHubMixin,
    TensorType,
    add_model_info_to_auto_map,
    add_model_info_to_custom_pipelines,
    cached_file,
    copy_func,
    direct_transformers_import,
    download_url,
    is_offline_mode,
    is_remote_url,
    logging,
)


logger = logging.get_logger(__name__)

# Dynamically import the Transformers module to grab the attribute classes of the processor form their names.
transformers_module = direct_transformers_import(Path(__file__).parent)


AUTO_TO_BASE_CLASS_MAPPING = {
    "AutoTokenizer": "PreTrainedTokenizerBase",
    "AutoFeatureExtractor": "FeatureExtractionMixin",
    "AutoImageProcessor": "ImageProcessingMixin",
}

if sys.version_info >= (3, 11):
    Unpack = typing.Unpack
else:
    Unpack = typing_extensions.Unpack


class TextKwargs(TypedDict, total=False):
    """
    Keyword arguments for text processing. For extended documentation, check out tokenization_utils_base methods and
    docstrings associated.

    Attributes:
        add_special_tokens (`bool`, *optional*)
            Whether or not to add special tokens when encoding the sequences.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*)
            Activates and controls padding.
        truncation (`bool`, `str` or [`~tokenization_utils_base.TruncationStrategy`], *optional*):
            Activates and controls truncation.
        max_length (`int`, *optional*):
            Controls the maximum length to use by one of the truncation/padding parameters.
        stride (`int`, *optional*):
            If set, the overflowing tokens will contain some tokens from the end of the truncated sequence.
        is_split_into_words (`bool`, *optional*):
            Whether or not the input is already pre-tokenized.
        pad_to_multiple_of (`int`, *optional*):
            If set, will pad the sequence to a multiple of the provided value.
        return_token_type_ids (`bool`, *optional*):
            Whether to return token type IDs.
        return_attention_mask (`bool`, *optional*):
            Whether to return the attention mask.
        return_overflowing_tokens (`bool`, *optional*):
            Whether or not to return overflowing token sequences.
        return_special_tokens_mask (`bool`, *optional*):
            Whether or not to return special tokens mask information.
        return_offsets_mapping (`bool`, *optional*):
            Whether or not to return `(char_start, char_end)` for each token.
        return_length (`bool`, *optional*):
            Whether or not to return the lengths of the encoded inputs.
        verbose (`bool`, *optional*):
            Whether or not to print more information and warnings.
        padding_side (`str`, *optional*):
            The side on which padding will be applied.
    """

    text_pair: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]]
    text_target: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]
    text_pair_target: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]]
    add_special_tokens: Optional[bool]
    padding: Union[bool, str, PaddingStrategy]
    truncation: Union[bool, str, TruncationStrategy]
    max_length: Optional[int]
    stride: Optional[int]
    is_split_into_words: Optional[bool]
    pad_to_multiple_of: Optional[int]
    return_token_type_ids: Optional[bool]
    return_attention_mask: Optional[bool]
    return_overflowing_tokens: Optional[bool]
    return_special_tokens_mask: Optional[bool]
    return_offsets_mapping: Optional[bool]
    return_length: Optional[bool]
    verbose: Optional[bool]
    padding_side: Optional[str]


class ImagesKwargs(TypedDict, total=False):
    """
    Keyword arguments for image processing. For extended documentation, check the appropriate ImageProcessor
    class methods and docstrings.

    Attributes:
        do_resize (`bool`, *optional*):
            Whether to resize the image.
        size (`Dict[str, int]`, *optional*):
            Resize the shorter side of the input to `size["shortest_edge"]`.
        size_divisor (`int`, *optional*):
            The size by which to make sure both the height and width can be divided.
        crop_size (`Dict[str, int]`, *optional*):
            Desired output size when applying center-cropping.
        resample (`PILImageResampling`, *optional*):
            Resampling filter to use if resizing the image.
        do_rescale (`bool`, *optional*):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*):
            Mean to use if normalizing the image.
        image_std (`float` or `List[float]`, *optional*):
            Standard deviation to use if normalizing the image.
        do_pad (`bool`, *optional*):
            Whether to pad the image to the `(max_height, max_width)` of the images in the batch.
        pad_size (`Dict[str, int]`, *optional*):
            The size `{"height": int, "width" int}` to pad the images to.
        do_center_crop (`bool`, *optional*):
            Whether to center crop the image.
        data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the output image.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input image.
        device (`str`, *optional*):
            The device to use for processing (e.g. "cpu", "cuda"), only relevant for fast image processing.
    """

    do_resize: Optional[bool]
    size: Optional[Dict[str, int]]
    size_divisor: Optional[int]
    crop_size: Optional[Dict[str, int]]
    resample: Optional[Union["PILImageResampling", int]]
    do_rescale: Optional[bool]
    rescale_factor: Optional[float]
    do_normalize: Optional[bool]
    image_mean: Optional[Union[float, List[float]]]
    image_std: Optional[Union[float, List[float]]]
    do_pad: Optional[bool]
    pad_size: Optional[Dict[str, int]]
    do_center_crop: Optional[bool]
    data_format: Optional[ChannelDimension]
    input_data_format: Optional[Union[str, ChannelDimension]]
    device: Optional[str]


class VideosKwargs(TypedDict, total=False):
    """
    Keyword arguments for video processing.

    Attributes:
        do_resize (`bool`):
            Whether to resize the image.
        size (`Dict[str, int]`, *optional*):
            Resize the shorter side of the input to `size["shortest_edge"]`.
        size_divisor (`int`, *optional*):
            The size by which to make sure both the height and width can be divided.
        resample (`PILImageResampling`, *optional*):
            Resampling filter to use if resizing the image.
        do_rescale (`bool`, *optional*):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*):
            Mean to use if normalizing the image.
        image_std (`float` or `List[float]`, *optional*):
            Standard deviation to use if normalizing the image.
        do_pad (`bool`, *optional*):
            Whether to pad the image to the `(max_height, max_width)` of the images in the batch.
        do_center_crop (`bool`, *optional*):
            Whether to center crop the image.
        data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the output image.
        input_data_format (`ChannelDimension` or `str`, *optional*):
            The channel dimension format for the input image.
    """

    do_resize: Optional[bool]
    size: Optional[Dict[str, int]]
    size_divisor: Optional[int]
    resample: Optional["PILImageResampling"]
    do_rescale: Optional[bool]
    rescale_factor: Optional[float]
    do_normalize: Optional[bool]
    image_mean: Optional[Union[float, List[float]]]
    image_std: Optional[Union[float, List[float]]]
    do_pad: Optional[bool]
    do_center_crop: Optional[bool]
    data_format: Optional[ChannelDimension]
    input_data_format: Optional[Union[str, ChannelDimension]]


class AudioKwargs(TypedDict, total=False):
    """
    Keyword arguments for audio processing.

    Attributes:
        sampling_rate (`int`, *optional*):
            The sampling rate at which the `raw_speech` input was sampled.
        raw_speech (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`):
            The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
            values, a list of numpy arrays or a list of list of float values. Must be mono channel audio, not
            stereo, i.e. single float per timestep.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding
            index) among:

            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                sequence if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        truncation (`bool`, *optional*):
            Activates truncation to cut input sequences longer than *max_length* to *max_length*.
        pad_to_multiple_of (`int`, *optional*):
            If set, will pad the sequence to a multiple of the provided value.
        return_attention_mask (`bool`, *optional*):
            Whether or not [`~ASTFeatureExtractor.__call__`] should return `attention_mask`.
    """

    sampling_rate: Optional[int]
    raw_speech: Optional[Union["np.ndarray", List[float], List["np.ndarray"], List[List[float]]]]
    padding: Optional[Union[bool, str, PaddingStrategy]]
    max_length: Optional[int]
    truncation: Optional[bool]
    pad_to_multiple_of: Optional[int]
    return_attention_mask: Optional[bool]


class CommonKwargs(TypedDict, total=False):
    return_tensors: Optional[Union[str, TensorType]]


class ProcessingKwargs(TextKwargs, ImagesKwargs, VideosKwargs, AudioKwargs, CommonKwargs, total=False):
    """
    Base class for kwargs passing to processors.
    A model should have its own `ModelProcessorKwargs` class that inherits from `ProcessingKwargs` to provide:
        1) Additional typed keys and that this model requires to process inputs.
        2) Default values for existing keys under a `_defaults` attribute.
    New keys have to be defined as follows to ensure type hinting is done correctly.

    ```python
    # adding a new image kwarg for this model
    class ModelImagesKwargs(ImagesKwargs, total=False):
        new_image_kwarg: Optional[bool]

    class ModelProcessorKwargs(ProcessingKwargs, total=False):
        images_kwargs: ModelImagesKwargs
        _defaults = {
            "images_kwargs: {
                "new_image_kwarg": False,
            }
            "text_kwargs": {
                "padding": "max_length",
            },
        }

    ```

    For Python 3.8 compatibility, when inheriting from this class and overriding one of the kwargs,
    you need to manually update the __annotations__ dictionary. This can be done as follows:

    ```python
    class CustomProcessorKwargs(ProcessingKwargs, total=False):
        images_kwargs: CustomImagesKwargs

    CustomProcessorKwargs.__annotations__["images_kwargs"] = CustomImagesKwargs  # python 3.8 compatibility
    ```python

    """

    common_kwargs: CommonKwargs = {
        **CommonKwargs.__annotations__,
    }
    text_kwargs: TextKwargs = {
        **TextKwargs.__annotations__,
    }
    images_kwargs: ImagesKwargs = {
        **ImagesKwargs.__annotations__,
    }
    videos_kwargs: VideosKwargs = {
        **VideosKwargs.__annotations__,
    }
    audio_kwargs: AudioKwargs = {
        **AudioKwargs.__annotations__,
    }


class ChatTemplateKwargs(TypedDict, total=False):
    """
    Keyword arguments for processor chat templates.

    tokenize (`bool`, *optional*, defaults to `False`):
        Whether to tokenize the output or not.
    return_dict (`bool`, defaults to `False`):
        Whether to return a dictionary with named outputs. Has no effect if tokenize is `False`.
    tools (`List[Dict]`, *optional*):
        A list of tools (callable functions) that will be accessible to the model. If the template does not
        support function calling, this argument will have no effect. Each tool should be passed as a JSON Schema,
        giving the name, description and argument types for the tool. See our
        [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#automated-function-conversion-for-tool-use)
        for more information.
    documents (`List[Dict[str, str]]`, *optional*):
        A list of dicts representing documents that will be accessible to the model if it is performing RAG
        (retrieval-augmented generation). If the template does not support RAG, this argument will have no
        effect. We recommend that each document should be a dict containing "title" and "text" keys. Please
        see the RAG section of the [chat templating guide](https://huggingface.co/docs/transformers/main/en/chat_templating#arguments-for-RAG)
        for examples of passing documents with chat templates.
    add_generation_prompt (bool, *optional*):
        If this is set, a prompt with the token(s) that indicate
        the start of an assistant message will be appended to the formatted output. This is useful when you want to generate a response from the model.
        Note that this argument will be passed to the chat template, and so it must be supported in the
        template for this argument to have any effect.
    continue_final_message (bool, *optional*):
        If this is set, the chat will be formatted so that the final
        message in the chat is open-ended, without any EOS tokens. The model will continue this message
        rather than starting a new one. This allows you to "prefill" part of
        the model's response for it. Cannot be used at the same time as `add_generation_prompt`.
    return_assistant_tokens_mask (`bool`, defaults to `False`):
        Whether to return a mask of the assistant generated tokens. For tokens generated by the assistant,
        the mask will contain 1. For user and system tokens, the mask will contain 0.
        This functionality is only available for chat templates that support it via the `{% generation %}` keyword.
    num_frames (`int`, *optional*):
        Number of frames to sample uniformly. If not passed, the whole video is loaded.
    video_load_backend (`str`, *optional*, defaults to `"pyav"`):
        The backend to use when loading the video which will be used only when there are videos in the conversation.
        Can be any of ["decord", "pyav", "opencv", "torchvision"]. Defaults to "pyav" because it is the only backend
        that supports all types of sources to load from.
    """

    tokenize: Optional[bool] = False
    return_dict: Optional[bool] = False
    tools: Optional[List[Dict]] = None
    documents: Optional[List[Dict[str, str]]] = None
    add_generation_prompt: Optional[bool] = False
    continue_final_message: Optional[bool] = False
    return_assistant_tokens_mask: Optional[bool] = False
    num_frames: Optional[int] = None
    video_load_backend: Optional[str] = "pyav"


class AllKwargsForChatTemplate(
    TextKwargs, ImagesKwargs, VideosKwargs, AudioKwargs, CommonKwargs, ChatTemplateKwargs
): ...


class ProcessorMixin(PushToHubMixin):
    """
    This is a mixin used to provide saving/loading functionality for all processor classes.
    """

    attributes = ["feature_extractor", "tokenizer"]
    optional_attributes = ["chat_template"]
    optional_call_args: List[str] = []
    # Names need to be attr_class for attr in attributes
    feature_extractor_class = None
    tokenizer_class = None
    _auto_class = None
    valid_kwargs: List[str] = []

    # args have to match the attributes class attribute
    def __init__(self, *args, **kwargs):
        # First, extract optional attributes from kwargs if present
        # Optional attributes can never be positional arguments
        for optional_attribute in self.optional_attributes:
            setattr(self, optional_attribute, kwargs.pop(optional_attribute, None))
        # Sanitize args and kwargs
        for key in kwargs:
            if key not in self.attributes:
                raise TypeError(f"Unexpected keyword argument {key}.")
        for arg, attribute_name in zip(args, self.attributes):
            if attribute_name in kwargs:
                raise TypeError(f"Got multiple values for argument {attribute_name}.")
            else:
                kwargs[attribute_name] = arg

        if len(kwargs) != len(self.attributes):
            raise ValueError(
                f"This processor requires {len(self.attributes)} arguments: {', '.join(self.attributes)}. Got "
                f"{len(args)} arguments instead."
            )

        # Check each arg is of the proper class (this will also catch a user initializing in the wrong order)
        for attribute_name, arg in kwargs.items():
            class_name = getattr(self, f"{attribute_name}_class")
            # Nothing is ever going to be an instance of "AutoXxx", in that case we check the base class.
            class_name = AUTO_TO_BASE_CLASS_MAPPING.get(class_name, class_name)
            if isinstance(class_name, tuple):
                proper_class = tuple(getattr(transformers_module, n) for n in class_name if n is not None)
            else:
                proper_class = getattr(transformers_module, class_name)

            if not isinstance(arg, proper_class):
                raise TypeError(
                    f"Received a {type(arg).__name__} for argument {attribute_name}, but a {class_name} was expected."
                )

            setattr(self, attribute_name, arg)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this processor instance.
        """
        output = copy.deepcopy(self.__dict__)

        # Get the kwargs in `__init__`.
        sig = inspect.signature(self.__init__)
        # Only save the attributes that are presented in the kwargs of `__init__`.
        attrs_to_save = sig.parameters
        # Don't save attributes like `tokenizer`, `image processor` etc.
        attrs_to_save = [x for x in attrs_to_save if x not in self.__class__.attributes]
        # extra attributes to be kept
        attrs_to_save += ["auto_map"]

        output = {k: v for k, v in output.items() if k in attrs_to_save}

        output["processor_class"] = self.__class__.__name__

        if "tokenizer" in output:
            del output["tokenizer"]
        if "image_processor" in output:
            del output["image_processor"]
        if "feature_extractor" in output:
            del output["feature_extractor"]
        if "chat_template" in output:
            del output["chat_template"]

        # Some attributes have different names but containing objects that are not simple strings
        output = {
            k: v
            for k, v in output.items()
            if not (isinstance(v, PushToHubMixin) or v.__class__.__name__ == "BeamSearchDecoderCTC")
        }

        return output

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this feature_extractor instance in JSON format.
        """
        dictionary = self.to_dict()

        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this processor instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def __repr__(self):
        attributes_repr = [f"- {name}: {repr(getattr(self, name))}" for name in self.attributes]
        attributes_repr = "\n".join(attributes_repr)
        return f"{self.__class__.__name__}:\n{attributes_repr}\n\n{self.to_json_string()}"

    def save_pretrained(self, save_directory, push_to_hub: bool = False, **kwargs):
        """
        Saves the attributes of this processor (feature extractor, tokenizer...) in the specified directory so that it
        can be reloaded using the [`~ProcessorMixin.from_pretrained`] method.

        <Tip>

        This class method is simply calling [`~feature_extraction_utils.FeatureExtractionMixin.save_pretrained`] and
        [`~tokenization_utils_base.PreTrainedTokenizerBase.save_pretrained`]. Please refer to the docstrings of the
        methods above for more information.

        </Tip>

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the feature extractor JSON file and the tokenizer files will be saved (directory will
                be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        use_auth_token = kwargs.pop("use_auth_token", None)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token", None) is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)
        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            attrs = [getattr(self, attribute_name) for attribute_name in self.attributes]
            configs = [(a.init_kwargs if isinstance(a, PreTrainedTokenizerBase) else a) for a in attrs]
            configs.append(self)
            custom_object_save(self, save_directory, config=configs)

        for attribute_name in self.attributes:
            attribute = getattr(self, attribute_name)
            # Include the processor class in the attribute config so this processor can then be reloaded with the
            # `AutoProcessor` API.
            if hasattr(attribute, "_set_processor_class"):
                attribute._set_processor_class(self.__class__.__name__)
            attribute.save_pretrained(save_directory)

        if self._auto_class is not None:
            # We added an attribute to the init_kwargs of the tokenizers, which needs to be cleaned up.
            for attribute_name in self.attributes:
                attribute = getattr(self, attribute_name)
                if isinstance(attribute, PreTrainedTokenizerBase):
                    del attribute.init_kwargs["auto_map"]

        # If we save using the predefined names, we can load using `from_pretrained`
        # plus we save chat_template in its own file
        output_processor_file = os.path.join(save_directory, PROCESSOR_NAME)
        output_raw_chat_template_file = os.path.join(save_directory, "chat_template.jinja")
        output_chat_template_file = os.path.join(save_directory, "chat_template.json")

        processor_dict = self.to_dict()
        # Save `chat_template` in its own file. We can't get it from `processor_dict` as we popped it in `to_dict`
        # to avoid serializing chat template in json config file. So let's get it from `self` directly
        if self.chat_template is not None:
            if kwargs.get("save_raw_chat_template", False):
                with open(output_raw_chat_template_file, "w", encoding="utf-8") as writer:
                    writer.write(self.chat_template)
                logger.info(f"chat template saved in {output_raw_chat_template_file}")
            else:
                chat_template_json_string = (
                    json.dumps({"chat_template": self.chat_template}, indent=2, sort_keys=True) + "\n"
                )
                with open(output_chat_template_file, "w", encoding="utf-8") as writer:
                    writer.write(chat_template_json_string)
                logger.info(f"chat template saved in {output_chat_template_file}")

        # For now, let's not save to `processor_config.json` if the processor doesn't have extra attributes and
        # `auto_map` is not specified.
        if set(processor_dict.keys()) != {"processor_class"}:
            self.to_json_file(output_processor_file)
            logger.info(f"processor saved in {output_processor_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

        if set(processor_dict.keys()) == {"processor_class"}:
            return []
        return [output_processor_file]

    @classmethod
    def get_processor_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        processor of type [`~processing_utils.ProcessingMixin`] using `from_args_and_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
            subfolder (`str`, *optional*, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
                specify the folder name here.

        Returns:
            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the processor object.
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")

        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {"file_type": "processor", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            processor_file = os.path.join(pretrained_model_name_or_path, PROCESSOR_NAME)

        if os.path.isfile(pretrained_model_name_or_path):
            resolved_processor_file = pretrained_model_name_or_path
            # cant't load chat-template when given a file as pretrained_model_name_or_path
            resolved_chat_template_file = None
            resolved_raw_chat_template_file = None
            is_local = True
        elif is_remote_url(pretrained_model_name_or_path):
            processor_file = pretrained_model_name_or_path
            resolved_processor_file = download_url(pretrained_model_name_or_path)
            # can't load chat-template when given a file url as pretrained_model_name_or_path
            resolved_chat_template_file = None
            resolved_raw_chat_template_file = None
        else:
            processor_file = PROCESSOR_NAME
            chat_template_file = "chat_template.json"
            raw_chat_template_file = "chat_template.jinja"
            try:
                # Load from local folder or from cache or download from model Hub and cache
                resolved_processor_file = cached_file(
                    pretrained_model_name_or_path,
                    processor_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                )

                # Load chat template from a separate json if exists
                # because making it part of processor-config break BC.
                # Processors in older version do not accept any kwargs
                resolved_chat_template_file = cached_file(
                    pretrained_model_name_or_path,
                    chat_template_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                )

                resolved_raw_chat_template_file = cached_file(
                    pretrained_model_name_or_path,
                    raw_chat_template_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_missing_entries=False,
                )
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load processor for '{pretrained_model_name_or_path}'. If you were trying to load"
                    " it from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a {PROCESSOR_NAME} file"
                )

        # Add chat template as kwarg before returning because most models don't have processor config
        if resolved_raw_chat_template_file is not None:
            with open(resolved_raw_chat_template_file, "r", encoding="utf-8") as reader:
                chat_template = reader.read()
            kwargs["chat_template"] = chat_template
        elif resolved_chat_template_file is not None:
            with open(resolved_chat_template_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            chat_template = json.loads(text)["chat_template"]
            kwargs["chat_template"] = chat_template

        # Existing processors on the Hub created before #27761 being merged don't have `processor_config.json` (if not
        # updated afterward), and we need to keep `from_pretrained` work. So here it fallbacks to the empty dict.
        # (`cached_file` called using `_raise_exceptions_for_missing_entries=False` to avoid exception)
        # However, for models added in the future, we won't get the expected error if this file is missing.
        if resolved_processor_file is None:
            return {}, kwargs

        try:
            # Load processor dict
            with open(resolved_processor_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            processor_dict = json.loads(text)

        except json.JSONDecodeError:
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_processor_file}' is not a valid JSON file."
            )

        if is_local:
            logger.info(f"loading configuration file {resolved_processor_file}")
        else:
            logger.info(f"loading configuration file {processor_file} from cache at {resolved_processor_file}")

        if "chat_template" in processor_dict and processor_dict["chat_template"] is not None:
            logger.warning_once(
                "Chat templates should be in a 'chat_template.jinja' file but found key='chat_template' "
                "in the processor's config. Make sure to move your template to its own file."
            )

        if not is_local:
            if "auto_map" in processor_dict:
                processor_dict["auto_map"] = add_model_info_to_auto_map(
                    processor_dict["auto_map"], pretrained_model_name_or_path
                )
            if "custom_pipelines" in processor_dict:
                processor_dict["custom_pipelines"] = add_model_info_to_custom_pipelines(
                    processor_dict["custom_pipelines"], pretrained_model_name_or_path
                )

        return processor_dict, kwargs

    @classmethod
    def from_args_and_dict(cls, args, processor_dict: Dict[str, Any], **kwargs):
        """
        Instantiates a type of [`~processing_utils.ProcessingMixin`] from a Python dictionary of parameters.

        Args:
            processor_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the processor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~processing_utils.ProcessingMixin.to_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the processor object.

        Returns:
            [`~processing_utils.ProcessingMixin`]: The processor object instantiated from those
            parameters.
        """
        processor_dict = processor_dict.copy()
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)
        chat_template = kwargs.pop("chat_template", None)

        # We have to pop up some unused (but specific) kwargs and then validate that it doesn't contain unused kwargs
        # If we don't pop, some specific kwargs will raise a warning
        if "processor_class" in processor_dict:
            del processor_dict["processor_class"]

        if "auto_map" in processor_dict:
            del processor_dict["auto_map"]

        unused_kwargs = cls.validate_init_kwargs(processor_config=processor_dict, valid_kwargs=cls.valid_kwargs)
        processor = cls(*args, **processor_dict)
        if chat_template is not None:
            setattr(processor, "chat_template", chat_template)

        # Update processor with kwargs if needed
        for key in set(kwargs.keys()):
            if hasattr(processor, key):
                setattr(processor, key, kwargs.pop(key))

        kwargs.update(unused_kwargs)
        logger.info(f"Processor {processor}")
        if return_unused_kwargs:
            return processor, kwargs
        else:
            return processor

    def _merge_kwargs(
        self,
        ModelProcessorKwargs: ProcessingKwargs,
        tokenizer_init_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Dict]:
        """
        Method to merge dictionaries of kwargs cleanly separated by modality within a Processor instance.
        The order of operations is as follows:
            1) kwargs passed as before have highest priority to preserve BC.
                ```python
                high_priority_kwargs = {"crop_size" = {"height": 222, "width": 222}, "padding" = "max_length"}
                processor(..., **high_priority_kwargs)
                ```
            2) kwargs passed as modality-specific kwargs have second priority. This is the recommended API.
                ```python
                processor(..., text_kwargs={"padding": "max_length"}, images_kwargs={"crop_size": {"height": 222, "width": 222}}})
                ```
            3) kwargs passed during instantiation of a modality processor have fourth priority.
                ```python
                tokenizer = tokenizer_class(..., {"padding": "max_length"})
                image_processor = image_processor_class(...)
                processor(tokenizer, image_processor) # will pass max_length unless overriden by kwargs at call
                ```
            4) defaults kwargs specified at processor level have lowest priority.
                ```python
                class MyProcessingKwargs(ProcessingKwargs, CommonKwargs, TextKwargs, ImagesKwargs, total=False):
                    _defaults = {
                        "text_kwargs": {
                            "padding": "max_length",
                            "max_length": 64,
                        },
                    }
                ```
        Args:
            ModelProcessorKwargs (`ProcessingKwargs`):
                Typed dictionary of kwargs specifically required by the model passed.
            tokenizer_init_kwargs (`Dict`, *optional*):
                Dictionary of kwargs the tokenizer was instantiated with and need to take precedence over defaults.

        Returns:
            output_kwargs (`Dict`):
                Dictionary of per-modality kwargs to be passed to each modality-specific processor.

        """
        # Initialize dictionaries
        output_kwargs = {
            "text_kwargs": {},
            "images_kwargs": {},
            "audio_kwargs": {},
            "videos_kwargs": {},
            "common_kwargs": {},
        }

        default_kwargs = {
            "text_kwargs": {},
            "images_kwargs": {},
            "audio_kwargs": {},
            "videos_kwargs": {},
            "common_kwargs": {},
        }

        used_keys = set()

        # get defaults from set model processor kwargs if they exist
        for modality in default_kwargs:
            default_kwargs[modality] = ModelProcessorKwargs._defaults.get(modality, {}).copy()
            # update defaults with arguments from tokenizer init
            for modality_key in ModelProcessorKwargs.__annotations__[modality].__annotations__.keys():
                # init with tokenizer init kwargs if necessary
                if modality_key in tokenizer_init_kwargs:
                    value = (
                        getattr(self.tokenizer, modality_key)
                        if hasattr(self.tokenizer, modality_key)
                        else tokenizer_init_kwargs[modality_key]
                    )
                    default_kwargs[modality][modality_key] = value
        # now defaults kwargs are updated with the tokenizers defaults.
        # pass defaults to output dictionary
        output_kwargs.update(default_kwargs)

        # update modality kwargs with passed kwargs
        non_modality_kwargs = set(kwargs) - set(output_kwargs)
        for modality in output_kwargs:
            for modality_key in ModelProcessorKwargs.__annotations__[modality].__annotations__.keys():
                # check if we received a structured kwarg dict or not to handle it correctly
                if modality in kwargs:
                    kwarg_value = kwargs[modality].pop(modality_key, "__empty__")
                    # check if this key was passed as a flat kwarg.
                    if kwarg_value != "__empty__" and modality_key in non_modality_kwargs:
                        raise ValueError(
                            f"Keyword argument {modality_key} was passed two times:\n"
                            f"in a dictionary for {modality} and as a **kwarg."
                        )
                elif modality_key in kwargs:
                    # we get a modality_key instead of popping it because modality-specific processors
                    # can have overlapping kwargs
                    kwarg_value = kwargs.get(modality_key, "__empty__")
                else:
                    kwarg_value = "__empty__"
                if kwarg_value != "__empty__":
                    output_kwargs[modality][modality_key] = kwarg_value
                    used_keys.add(modality_key)

        # Determine if kwargs is a flat dictionary or contains nested dictionaries
        if any(key in default_kwargs for key in kwargs):
            # kwargs is dictionary-based, and some keys match modality names
            for modality, subdict in kwargs.items():
                if modality in default_kwargs:
                    for subkey, subvalue in subdict.items():
                        if subkey not in used_keys:
                            output_kwargs[modality][subkey] = subvalue
                            used_keys.add(subkey)
        else:
            # kwargs is a flat dictionary
            for key in kwargs:
                if key not in used_keys:
                    if key in ModelProcessorKwargs.__annotations__["common_kwargs"].__annotations__.keys():
                        output_kwargs["common_kwargs"][key] = kwargs[key]
                    else:
                        logger.warning_once(
                            f"Keyword argument `{key}` is not a valid argument for this processor and will be ignored."
                        )

        # all modality-specific kwargs are updated with common kwargs
        for modality in output_kwargs:
            output_kwargs[modality].update(output_kwargs["common_kwargs"])
        return output_kwargs

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ):
        r"""
        Instantiate a processor associated with a pretrained model.

        <Tip>

        This class method is simply calling the feature extractor
        [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`], image processor
        [`~image_processing_utils.ImageProcessingMixin`] and the tokenizer
        [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`] methods. Please refer to the docstrings of the
        methods above for more information.

        </Tip>

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a feature extractor file saved using the
                  [`~SequenceFeatureExtractor.save_pretrained`] method, e.g., `./my_model_directory/`.
                - a path or url to a saved feature extractor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            **kwargs
                Additional keyword arguments passed along to both
                [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`] and
                [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`].
        """
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision

        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None:
            kwargs["token"] = token

        args = cls._get_arguments_from_pretrained(pretrained_model_name_or_path, **kwargs)
        processor_dict, kwargs = cls.get_processor_dict(pretrained_model_name_or_path, **kwargs)

        return cls.from_args_and_dict(args, processor_dict, **kwargs)

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoProcessor"):
        """
        Register this class with a given auto class. This should only be used for custom feature extractors as the ones
        in the library are already mapped with `AutoProcessor`.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoProcessor"`):
                The auto class to register this new feature extractor with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class

    @classmethod
    def _get_arguments_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        args = []
        for attribute_name in cls.attributes:
            class_name = getattr(cls, f"{attribute_name}_class")
            if isinstance(class_name, tuple):
                classes = tuple(getattr(transformers_module, n) if n is not None else None for n in class_name)
                use_fast = kwargs.get("use_fast", True)
                if use_fast and classes[1] is not None:
                    attribute_class = classes[1]
                else:
                    attribute_class = classes[0]
            else:
                attribute_class = getattr(transformers_module, class_name)

            args.append(attribute_class.from_pretrained(pretrained_model_name_or_path, **kwargs))
        return args

    @property
    def model_input_names(self):
        first_attribute = getattr(self, self.attributes[0])
        return getattr(first_attribute, "model_input_names", None)

    @staticmethod
    def validate_init_kwargs(processor_config, valid_kwargs):
        kwargs_from_config = processor_config.keys()
        unused_kwargs = {}
        unused_keys = set(kwargs_from_config) - set(valid_kwargs)
        if unused_keys:
            unused_key_str = ", ".join(unused_keys)
            logger.warning(
                f"Some kwargs in processor config are unused and will not have any effect: {unused_key_str}. "
            )
            unused_kwargs = {k: processor_config[k] for k in unused_keys}
        return unused_kwargs

    def prepare_and_validate_optional_call_args(self, *args):
        """
        Matches optional positional arguments to their corresponding names in `optional_call_args`
        in the processor class in the order they are passed to the processor call.

        Note that this should only be used in the `__call__` method of the processors with special
        arguments. Special arguments are arguments that aren't `text`, `images`, `audio`, nor `videos`
        but also aren't passed to the tokenizer, image processor, etc. Examples of such processors are:
            - `CLIPSegProcessor`
            - `LayoutLMv2Processor`
            - `OwlViTProcessor`

        Also note that passing by position to the processor call is now deprecated and will be disallowed
        in future versions. We only have this for backward compatibility.

        Example:
            Suppose that the processor class has `optional_call_args = ["arg_name_1", "arg_name_2"]`.
            And we define the call method as:
            ```python
            def __call__(
                self,
                text: str,
                images: Optional[ImageInput] = None,
                *arg,
                audio=None,
                videos=None,
            )
            ```

            Then, if we call the processor as:
            ```python
            images = [...]
            processor("What is common in these images?", images, arg_value_1, arg_value_2)
            ```

            Then, this method will return:
            ```python
            {
                "arg_name_1": arg_value_1,
                "arg_name_2": arg_value_2,
            }
            ```
            which we could then pass as kwargs to `self._merge_kwargs`
        """
        if len(args):
            warnings.warn(
                "Passing positional arguments to the processor call is now deprecated and will be disallowed in v4.47. "
                "Please pass all arguments as keyword arguments."
            )
        if len(args) > len(self.optional_call_args):
            raise ValueError(
                f"Expected *at most* {len(self.optional_call_args)} optional positional arguments in processor call"
                f"which will be matched with {' '.join(self.optional_call_args)} in the order they are passed."
                f"However, got {len(args)} positional arguments instead."
                "Please pass all arguments as keyword arguments instead (e.g. `processor(arg_name_1=..., arg_name_2=...))`."
            )
        return {arg_name: arg_value for arg_value, arg_name in zip(args, self.optional_call_args)}

    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]]],
        chat_template: Optional[str] = None,
        **kwargs: Unpack[AllKwargsForChatTemplate],
    ) -> str:
        """
        Similar to the `apply_chat_template` method on tokenizers, this method applies a Jinja template to input
        conversations to turn them into a single tokenizable string.

        The input is expected to be in the following format, where each message content is a list consisting of text and
        optionally image or video inputs. One can also provide an image, video, URL or local path which will be used to form
        `pixel_values` when `return_dict=True`. If not provided, one will get only the formatted text, optionally tokenized text.

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "https://www.ilankelman.org/stopsigns/australia.jpg"},
                    {"type": "text", "text": "Please describe this image in detail."},
                ],
            },
        ]

        Args:
            conversation (`List[Dict, str, str]`):
                The conversation to format.
            chat_template (`Optional[str]`, *optional*):
                The Jinja template to use for formatting the conversation. If not provided, the tokenizer's
                chat template is used.
        """

        if chat_template is None:
            if self.chat_template is not None:
                chat_template = self.chat_template
            else:
                raise ValueError(
                    "No chat template is set for this processor. Please either set the `chat_template` attribute, "
                    "or provide a chat template as an argument. See "
                    "https://huggingface.co/docs/transformers/main/en/chat_templating for more information."
                )

        text_kwargs = {}
        for key in TextKwargs.__annotations__.keys():
            value = kwargs.pop(key, None)
            if value is not None:
                text_kwargs[key] = value

        chat_template_kwargs = {}
        for key in ChatTemplateKwargs.__annotations__.keys():
            value = kwargs.pop(key, getattr(ChatTemplateKwargs, key))
            chat_template_kwargs[key] = value

        # Pop kwargs that should not be used by tokenizer's `apply_chat_template`
        tokenize = chat_template_kwargs.pop("tokenize")
        return_dict = chat_template_kwargs.pop("return_dict")
        num_frames = chat_template_kwargs.pop("num_frames")
        video_load_backend = chat_template_kwargs.pop("video_load_backend")

        prompt = self.tokenizer.apply_chat_template(
            conversation,
            chat_template=chat_template,
            tokenize=False,
            return_dict=False,
            **text_kwargs,
            **chat_template_kwargs,
        )

        # we will have to return all processed inputs in a dict
        if tokenize:
            images, videos = [], []
            for message in conversation:
                visuals = [content for content in message["content"] if content["type"] in ["image", "video"]]
                for vision_info in visuals:
                    if vision_info["type"] == "image":
                        for key in ["image", "url", "path", "base64"]:
                            if key in vision_info:
                                images.append(load_image(vision_info[key]))
                    elif vision_info["type"] == "video":
                        for key in ["video", "url", "path"]:
                            if key in vision_info:
                                videos.append(
                                    load_video(vision_info[key], num_frames=num_frames, backend=video_load_backend)
                                )

            out = self(
                text=prompt,
                images=images if images else None,
                videos=videos if videos else None,
                **kwargs,
            )
            if return_dict:
                return out
            else:
                return out["input_ids"]
        return prompt

    def post_process_image_text_to_text(self, generated_outputs):
        """
        Post-process the output of a vlm to decode the text.

        Args:
            generated_outputs (`torch.Tensor` or `np.ndarray`):
                The output of the model `generate` function. The output is expected to be a tensor of shape `(batch_size, sequence_length)`
                or `(sequence_length,)`.

        Returns:
            `List[str]`: The decoded text.
        """
        return self.tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)


def _validate_images_text_input_order(images, text):
    """
    For backward compatibility: reverse the order of `images` and `text` inputs if they are swapped.
    This method should only be called for processors where `images` and `text` have been swapped for uniformization purposes.
    Note that this method assumes that two `None` inputs are valid inputs. If this is not the case, it should be handled
    in the processor's `__call__` method before calling this method.
    """

    def is_url(val) -> bool:
        return isinstance(val, str) and val.startswith("http")

    def _is_valid_images_input_for_processor(imgs):
        # If we have an list of images, make sure every image is valid
        if isinstance(imgs, (list, tuple)):
            for img in imgs:
                if not _is_valid_images_input_for_processor(img):
                    return False
        # If not a list or tuple, we have been given a single image or batched tensor of images
        elif not (is_valid_image(imgs) or is_url(imgs)):
            return False
        return True

    def _is_valid_text_input_for_processor(t):
        if isinstance(t, str):
            # Strings are fine
            return True
        elif isinstance(t, (list, tuple)):
            # List are fine as long as they are...
            if len(t) == 0:
                # ... not empty
                return False
            for t_s in t:
                return _is_valid_text_input_for_processor(t_s)
        return False

    def _is_valid(input, validator):
        return validator(input) or input is None

    images_is_valid = _is_valid(images, _is_valid_images_input_for_processor)
    images_is_text = _is_valid_text_input_for_processor(images)

    text_is_valid = _is_valid(text, _is_valid_text_input_for_processor)
    text_is_images = _is_valid_images_input_for_processor(text)
    # Handle cases where both inputs are valid
    if images_is_valid and text_is_valid:
        return images, text

    # Handle cases where inputs need to and can be swapped
    if (images is None and text_is_images) or (text is None and images_is_text) or (images_is_text and text_is_images):
        logger.warning_once(
            "You may have used the wrong order for inputs. `images` should be passed before `text`. "
            "The `images` and `text` inputs will be swapped. This behavior will be deprecated in transformers v4.47."
        )
        return text, images

    raise ValueError("Invalid input type. Check that `images` and/or `text` are valid inputs.")


ProcessorMixin.push_to_hub = copy_func(ProcessorMixin.push_to_hub)
if ProcessorMixin.push_to_hub.__doc__ is not None:
    ProcessorMixin.push_to_hub.__doc__ = ProcessorMixin.push_to_hub.__doc__.format(
        object="processor", object_class="AutoProcessor", object_files="processor files"
    )
