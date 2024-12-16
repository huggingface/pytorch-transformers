# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
Feature extraction saving/loading class for common feature extractors.
"""

import copy
import json
import os
import warnings
from collections import UserDict
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import numpy as np

from .dynamic_module_utils import custom_object_save
from .utils import (
    FEATURE_EXTRACTOR_NAME,
    PushToHubMixin,
    TensorType,
    add_model_info_to_auto_map,
    add_model_info_to_custom_pipelines,
    cached_file,
    copy_func,
    download_url,
    is_flax_available,
    is_jax_tensor,
    is_numpy_array,
    is_offline_mode,
    is_remote_url,
    is_tf_available,
    is_torch_available,
    is_torch_device,
    is_torch_dtype,
    logging,
    requires_backends,
)


if TYPE_CHECKING:
    if is_torch_available():
        import torch  # noqa


logger = logging.get_logger(__name__)

PreTrainedFeatureExtractor = Union["SequenceFeatureExtractor"]  # noqa: F821


class BatchFeature(UserDict):
    r"""
    Holds the output of the [`~SequenceFeatureExtractor.pad`] and feature extractor specific `__call__` methods.

    This class is derived from a python dictionary and can be used as a dictionary.

    Args:
        data (`dict`, *optional*):
            Dictionary of lists/arrays/tensors returned by the __call__/pad methods ('input_values', 'attention_mask',
            etc.).
        tensor_type (`Union[None, str, TensorType]`, *optional*):
            You can give a tensor_type here to convert the lists of integers in PyTorch/TensorFlow/Numpy Tensors at
            initialization.
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None, tensor_type: Union[None, str, TensorType] = None):
        super().__init__(data)
        self.convert_to_tensors(tensor_type=tensor_type)

    def __getitem__(self, item: str) -> Union[Any]:
        """
        If the key is a string, returns the value of the dict associated to `key` ('input_values', 'attention_mask',
        etc.).
        """
        if isinstance(item, str):
            return self.data[item]
        else:
            raise KeyError("Indexing with integers is not available when using Python based feature extractors")

    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError

    def __getstate__(self):
        return {"data": self.data}

    def __setstate__(self, state):
        if "data" in state:
            self.data = state["data"]

    # Copied from transformers.tokenization_utils_base.BatchEncoding.keys
    def keys(self):
        return self.data.keys()

    # Copied from transformers.tokenization_utils_base.BatchEncoding.values
    def values(self):
        return self.data.values()

    # Copied from transformers.tokenization_utils_base.BatchEncoding.items
    def items(self):
        return self.data.items()

    def _get_is_as_tensor_fns(self, tensor_type: Optional[Union[str, TensorType]] = None):
        if tensor_type is None:
            return None, None

        # Convert to TensorType
        if not isinstance(tensor_type, TensorType):
            tensor_type = TensorType(tensor_type)

        # Get a function reference for the correct framework
        if tensor_type == TensorType.TENSORFLOW:
            if not is_tf_available():
                raise ImportError(
                    "Unable to convert output to TensorFlow tensors format, TensorFlow is not installed."
                )
            import tensorflow as tf

            as_tensor = tf.constant
            is_tensor = tf.is_tensor
        elif tensor_type == TensorType.PYTORCH:
            if not is_torch_available():
                raise ImportError("Unable to convert output to PyTorch tensors format, PyTorch is not installed.")
            import torch  # noqa

            def as_tensor(value):
                if isinstance(value, (list, tuple)) and len(value) > 0:
                    if isinstance(value[0], np.ndarray):
                        value = np.array(value)
                    elif (
                        isinstance(value[0], (list, tuple))
                        and len(value[0]) > 0
                        and isinstance(value[0][0], np.ndarray)
                    ):
                        value = np.array(value)
                if isinstance(value, np.ndarray):
                    return torch.from_numpy(value)
                else:
                    return torch.tensor(value)

            is_tensor = torch.is_tensor
        elif tensor_type == TensorType.JAX:
            if not is_flax_available():
                raise ImportError("Unable to convert output to JAX tensors format, JAX is not installed.")
            import jax.numpy as jnp  # noqa: F811

            as_tensor = jnp.array
            is_tensor = is_jax_tensor
        else:

            def as_tensor(value, dtype=None):
                if isinstance(value, (list, tuple)) and isinstance(value[0], (list, tuple, np.ndarray)):
                    value_lens = [len(val) for val in value]
                    if len(set(value_lens)) > 1 and dtype is None:
                        # we have a ragged list so handle explicitly
                        value = as_tensor([np.asarray(val) for val in value], dtype=object)
                return np.asarray(value, dtype=dtype)

            is_tensor = is_numpy_array
        return is_tensor, as_tensor

    def convert_to_tensors(self, tensor_type: Optional[Union[str, TensorType]] = None):
        """
        Convert the inner content to tensors.

        Args:
            tensor_type (`str` or [`~utils.TensorType`], *optional*):
                The type of tensors to use. If `str`, should be one of the values of the enum [`~utils.TensorType`]. If
                `None`, no modification is done.
        """
        if tensor_type is None:
            return self

        is_tensor, as_tensor = self._get_is_as_tensor_fns(tensor_type)

        # Do the tensor conversion in batch
        for key, value in self.items():
            try:
                if not is_tensor(value):
                    tensor = as_tensor(value)

                    self[key] = tensor
            except:  # noqa E722
                if key == "overflowing_values":
                    raise ValueError("Unable to create tensor returning overflowing values of different lengths. ")
                raise ValueError(
                    "Unable to create tensor, you should probably activate padding "
                    "with 'padding=True' to have batched tensors with the same length."
                )

        return self

    def to(self, *args, **kwargs) -> "BatchFeature":
        """
        Send all values to device by calling `v.to(*args, **kwargs)` (PyTorch only). This should support casting in
        different `dtypes` and sending the `BatchFeature` to a different `device`.

        Args:
            args (`Tuple`):
                Will be passed to the `to(...)` function of the tensors.
            kwargs (`Dict`, *optional*):
                Will be passed to the `to(...)` function of the tensors.
                To enable asynchronous data transfer, set the `non_blocking` flag in `kwargs` (defaults to `False`).

        Returns:
            [`BatchFeature`]: The same instance after modification.
        """
        requires_backends(self, ["torch"])
        import torch  # noqa

        new_data = {}
        device = kwargs.get("device")
        non_blocking = kwargs.get("non_blocking", False)
        # Check if the args are a device or a dtype
        if device is None and len(args) > 0:
            # device should be always the first argument
            arg = args[0]
            if is_torch_dtype(arg):
                # The first argument is a dtype
                pass
            elif isinstance(arg, str) or is_torch_device(arg) or isinstance(arg, int):
                device = arg
            else:
                # it's something else
                raise ValueError(f"Attempting to cast a BatchFeature to type {str(arg)}. This is not supported.")
        # We cast only floating point tensors to avoid issues with tokenizers casting `LongTensor` to `FloatTensor`
        for k, v in self.items():
            # check if v is a floating point
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                # cast and send to device
                new_data[k] = v.to(*args, **kwargs)
            elif isinstance(v, torch.Tensor) and device is not None:
                new_data[k] = v.to(device=device, non_blocking=non_blocking)
            else:
                new_data[k] = v
        self.data = new_data
        return self


class FeatureExtractionMixin(PushToHubMixin):
    """
    This is a feature extraction mixin used to provide saving/loading functionality for sequential and image feature
    extractors.
    """

    _auto_class = None

    def __init__(self, **kwargs):
        """Set elements of `kwargs` as attributes."""
        # Pop "processor_class" as it should be saved as private attribute
        self._processor_class = kwargs.pop("processor_class", None)
        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    def _set_processor_class(self, processor_class: str):
        """Sets processor class as an attribute."""
        self._processor_class = processor_class

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
        Instantiate a type of [`~feature_extraction_utils.FeatureExtractionMixin`] from a feature extractor, *e.g.* a
        derived class of [`SequenceFeatureExtractor`].

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a feature extractor file saved using the
                  [`~feature_extraction_utils.FeatureExtractionMixin.save_pretrained`] method, e.g.,
                  `./my_model_directory/`.
                - a path or url to a saved feature extractor JSON *file*, e.g.,
                  `./my_model_directory/preprocessor_config.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model feature extractor should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the feature extractor files and override the cached versions
                if they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.


                <Tip>

                To test a pull request you made on the Hub, you can pass `revision="refs/pr/<pr_number>"`.

                </Tip>

            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final feature extractor object. If `True`, then this
                functions returns a `Tuple(feature_extractor, unused_kwargs)` where *unused_kwargs* is a dictionary
                consisting of the key/value pairs whose keys are not feature extractor attributes: i.e., the part of
                `kwargs` which has not been used to update `feature_extractor` and is otherwise ignored.
            kwargs (`Dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are feature extractor attributes will be used to override the
                loaded values. Behavior concerning key/value pairs whose keys are *not* feature extractor attributes is
                controlled by the `return_unused_kwargs` keyword parameter.

        Returns:
            A feature extractor of type [`~feature_extraction_utils.FeatureExtractionMixin`].

        Examples:

        ```python
        # We can't instantiate directly the base class *FeatureExtractionMixin* nor *SequenceFeatureExtractor* so let's show the examples on a
        # derived class: *Wav2Vec2FeatureExtractor*
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base-960h"
        )  # Download feature_extraction_config from huggingface.co and cache.
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "./test/saved_model/"
        )  # E.g. feature_extractor (or model) was saved using *save_pretrained('./test/saved_model/')*
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("./test/saved_model/preprocessor_config.json")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base-960h", return_attention_mask=False, foo=False
        )
        assert feature_extractor.return_attention_mask is False
        feature_extractor, unused_kwargs = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-base-960h", return_attention_mask=False, foo=False, return_unused_kwargs=True
        )
        assert feature_extractor.return_attention_mask is False
        assert unused_kwargs == {"foo": False}
        ```"""
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

        feature_extractor_dict, kwargs = cls.get_feature_extractor_dict(pretrained_model_name_or_path, **kwargs)

        return cls.from_dict(feature_extractor_dict, **kwargs)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save a feature_extractor object to the directory `save_directory`, so that it can be re-loaded using the
        [`~feature_extraction_utils.FeatureExtractionMixin.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the feature extractor JSON file will be saved (will be created if it does not exist).
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

        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_feature_extractor_file = os.path.join(save_directory, FEATURE_EXTRACTOR_NAME)

        self.to_json_file(output_feature_extractor_file)
        logger.info(f"Feature extractor saved in {output_feature_extractor_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

        return [output_feature_extractor_file]

    @classmethod
    def get_feature_extractor_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        feature extractor of type [`~feature_extraction_utils.FeatureExtractionMixin`] using `from_dict`.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.

        Returns:
            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the feature extractor object.
        """
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        subfolder = kwargs.pop("subfolder", None)
        token = kwargs.pop("token", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)

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

        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {"file_type": "feature extractor", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            feature_extractor_file = os.path.join(pretrained_model_name_or_path, FEATURE_EXTRACTOR_NAME)
        if os.path.isfile(pretrained_model_name_or_path):
            resolved_feature_extractor_file = pretrained_model_name_or_path
            is_local = True
        elif is_remote_url(pretrained_model_name_or_path):
            feature_extractor_file = pretrained_model_name_or_path
            resolved_feature_extractor_file = download_url(pretrained_model_name_or_path)
        else:
            feature_extractor_file = FEATURE_EXTRACTOR_NAME
            try:
                # Load from local folder or from cache or download from model Hub and cache
                resolved_feature_extractor_file = cached_file(
                    pretrained_model_name_or_path,
                    feature_extractor_file,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    local_files_only=local_files_only,
                    subfolder=subfolder,
                    token=token,
                    user_agent=user_agent,
                    revision=revision,
                )
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load feature extractor for '{pretrained_model_name_or_path}'. If you were trying to load"
                    " it from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a {FEATURE_EXTRACTOR_NAME} file"
                )

        try:
            # Load feature_extractor dict
            with open(resolved_feature_extractor_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            feature_extractor_dict = json.loads(text)

        except json.JSONDecodeError:
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_feature_extractor_file}' is not a valid JSON file."
            )

        if is_local:
            logger.info(f"loading configuration file {resolved_feature_extractor_file}")
        else:
            logger.info(
                f"loading configuration file {feature_extractor_file} from cache at {resolved_feature_extractor_file}"
            )

        if not is_local:
            if "auto_map" in feature_extractor_dict:
                feature_extractor_dict["auto_map"] = add_model_info_to_auto_map(
                    feature_extractor_dict["auto_map"], pretrained_model_name_or_path
                )
            if "custom_pipelines" in feature_extractor_dict:
                feature_extractor_dict["custom_pipelines"] = add_model_info_to_custom_pipelines(
                    feature_extractor_dict["custom_pipelines"], pretrained_model_name_or_path
                )

        return feature_extractor_dict, kwargs

    @classmethod
    def from_dict(cls, feature_extractor_dict: Dict[str, Any], **kwargs) -> PreTrainedFeatureExtractor:
        """
        Instantiates a type of [`~feature_extraction_utils.FeatureExtractionMixin`] from a Python dictionary of
        parameters.

        Args:
            feature_extractor_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the feature extractor object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~feature_extraction_utils.FeatureExtractionMixin.to_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the feature extractor object.

        Returns:
            [`~feature_extraction_utils.FeatureExtractionMixin`]: The feature extractor object instantiated from those
            parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        # Update feature_extractor with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if key in feature_extractor_dict:
                feature_extractor_dict[key] = value
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        feature_extractor = cls(**feature_extractor_dict)

        logger.info(f"Feature extractor {feature_extractor}")
        if return_unused_kwargs:
            return feature_extractor, kwargs
        else:
            return feature_extractor

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary. Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        output["feature_extractor_type"] = self.__class__.__name__
        if "mel_filters" in output:
            del output["mel_filters"]
        if "window" in output:
            del output["window"]
        return output

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]) -> PreTrainedFeatureExtractor:
        """
        Instantiates a feature extractor of type [`~feature_extraction_utils.FeatureExtractionMixin`] from the path to
        a JSON file of parameters.

        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.

        Returns:
            A feature extractor of type [`~feature_extraction_utils.FeatureExtractionMixin`]: The feature_extractor
            object instantiated from that JSON file.
        """
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        feature_extractor_dict = json.loads(text)
        return cls(**feature_extractor_dict)

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this feature_extractor instance in JSON format.
        """
        dictionary = self.to_dict()

        for key, value in dictionary.items():
            if isinstance(value, np.ndarray):
                dictionary[key] = value.tolist()

        # make sure private name "_processor_class" is correctly
        # saved as "processor_class"
        _processor_class = dictionary.pop("_processor_class", None)
        if _processor_class is not None:
            dictionary["processor_class"] = _processor_class

        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this feature_extractor instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoFeatureExtractor"):
        """
        Register this class with a given auto class. This should only be used for custom feature extractors as the ones
        in the library are already mapped with `AutoFeatureExtractor`.

        <Tip warning={true}>

        This API is experimental and may have some slight breaking changes in the next releases.

        </Tip>

        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoFeatureExtractor"`):
                The auto class to register this new feature extractor with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class


FeatureExtractionMixin.push_to_hub = copy_func(FeatureExtractionMixin.push_to_hub)
if FeatureExtractionMixin.push_to_hub.__doc__ is not None:
    FeatureExtractionMixin.push_to_hub.__doc__ = FeatureExtractionMixin.push_to_hub.__doc__.format(
        object="feature extractor", object_class="AutoFeatureExtractor", object_files="feature extractor file"
    )
