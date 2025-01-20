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
"""AutoFeatureExtractor class."""

import importlib
import json
import os
import warnings
from collections import OrderedDict
from typing import Dict, Optional, Union

# Build the list of all feature extractors
from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...feature_extraction_utils import FeatureExtractionMixin
from ...utils import CONFIG_NAME, FEATURE_EXTRACTOR_NAME, get_file_from_repo, logging
from .auto_factory import _LazyAutoMapping
from .configuration_auto import (
    CONFIG_MAPPING_NAMES,
    AutoConfig,
    model_type_to_module_name,
    replace_list_option_in_docstrings,
)


logger = logging.get_logger(__name__)

FEATURE_EXTRACTOR_MAPPING_NAMES = OrderedDict(
    [
        ("audio-spectrogram-transformer", "ASTFeatureExtractor"),
        ("beit", "BeitFeatureExtractor"),
        ("chinese_clip", "ChineseCLIPFeatureExtractor"),
        ("clap", "ClapFeatureExtractor"),
        ("clip", "CLIPFeatureExtractor"),
        ("clipseg", "ViTFeatureExtractor"),
        ("clvp", "ClvpFeatureExtractor"),
        ("conditional_detr", "ConditionalDetrFeatureExtractor"),
        ("convnext", "ConvNextFeatureExtractor"),
        ("cvt", "ConvNextFeatureExtractor"),
        ("dac", "DacFeatureExtractor"),
        ("data2vec-audio", "Wav2Vec2FeatureExtractor"),
        ("data2vec-vision", "BeitFeatureExtractor"),
        ("deformable_detr", "DeformableDetrFeatureExtractor"),
        ("deit", "DeiTFeatureExtractor"),
        ("detr", "DetrFeatureExtractor"),
        ("dinat", "ViTFeatureExtractor"),
        ("donut-swin", "DonutFeatureExtractor"),
        ("dpt", "DPTFeatureExtractor"),
        ("encodec", "EncodecFeatureExtractor"),
        ("flava", "FlavaFeatureExtractor"),
        ("glpn", "GLPNFeatureExtractor"),
        ("groupvit", "CLIPFeatureExtractor"),
        ("hubert", "Wav2Vec2FeatureExtractor"),
        ("imagegpt", "ImageGPTFeatureExtractor"),
        ("layoutlmv2", "LayoutLMv2FeatureExtractor"),
        ("layoutlmv3", "LayoutLMv3FeatureExtractor"),
        ("levit", "LevitFeatureExtractor"),
        ("maskformer", "MaskFormerFeatureExtractor"),
        ("mctct", "MCTCTFeatureExtractor"),
        ("mimi", "EncodecFeatureExtractor"),
        ("mobilenet_v1", "MobileNetV1FeatureExtractor"),
        ("mobilenet_v2", "MobileNetV2FeatureExtractor"),
        ("mobilevit", "MobileViTFeatureExtractor"),
        ("moonshine", "Wav2Vec2FeatureExtractor"),
        ("moshi", "EncodecFeatureExtractor"),
        ("nat", "ViTFeatureExtractor"),
        ("owlvit", "OwlViTFeatureExtractor"),
        ("perceiver", "PerceiverFeatureExtractor"),
        ("poolformer", "PoolFormerFeatureExtractor"),
        ("pop2piano", "Pop2PianoFeatureExtractor"),
        ("regnet", "ConvNextFeatureExtractor"),
        ("resnet", "ConvNextFeatureExtractor"),
        ("seamless_m4t", "SeamlessM4TFeatureExtractor"),
        ("seamless_m4t_v2", "SeamlessM4TFeatureExtractor"),
        ("segformer", "SegformerFeatureExtractor"),
        ("sew", "Wav2Vec2FeatureExtractor"),
        ("sew-d", "Wav2Vec2FeatureExtractor"),
        ("speech_to_text", "Speech2TextFeatureExtractor"),
        ("speecht5", "SpeechT5FeatureExtractor"),
        ("swiftformer", "ViTFeatureExtractor"),
        ("swin", "ViTFeatureExtractor"),
        ("swinv2", "ViTFeatureExtractor"),
        ("table-transformer", "DetrFeatureExtractor"),
        ("timesformer", "VideoMAEFeatureExtractor"),
        ("timm_wrapper", "TimmWrapperImageProcessor"),
        ("tvlt", "TvltFeatureExtractor"),
        ("unispeech", "Wav2Vec2FeatureExtractor"),
        ("unispeech-sat", "Wav2Vec2FeatureExtractor"),
        ("univnet", "UnivNetFeatureExtractor"),
        ("van", "ConvNextFeatureExtractor"),
        ("videomae", "VideoMAEFeatureExtractor"),
        ("vilt", "ViltFeatureExtractor"),
        ("vit", "ViTFeatureExtractor"),
        ("vit_mae", "ViTFeatureExtractor"),
        ("vit_msn", "ViTFeatureExtractor"),
        ("wav2vec2", "Wav2Vec2FeatureExtractor"),
        ("wav2vec2-bert", "Wav2Vec2FeatureExtractor"),
        ("wav2vec2-conformer", "Wav2Vec2FeatureExtractor"),
        ("wavlm", "Wav2Vec2FeatureExtractor"),
        ("whisper", "WhisperFeatureExtractor"),
        ("xclip", "CLIPFeatureExtractor"),
        ("yolos", "YolosFeatureExtractor"),
    ]
)

FEATURE_EXTRACTOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, FEATURE_EXTRACTOR_MAPPING_NAMES)


def feature_extractor_class_from_name(class_name: str):
    for module_name, extractors in FEATURE_EXTRACTOR_MAPPING_NAMES.items():
        if class_name in extractors:
            module_name = model_type_to_module_name(module_name)

            module = importlib.import_module(f".{module_name}", "transformers.models")
            try:
                return getattr(module, class_name)
            except AttributeError:
                continue

    for _, extractor in FEATURE_EXTRACTOR_MAPPING._extra_content.items():
        if getattr(extractor, "__name__", None) == class_name:
            return extractor

    # We did not fine the class, but maybe it's because a dep is missing. In that case, the class will be in the main
    # init and we return the proper dummy to get an appropriate error message.
    main_module = importlib.import_module("transformers")
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)

    return None


def get_feature_extractor_config(
    pretrained_model_name_or_path: Union[str, os.PathLike],
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: Optional[bool] = None,
    proxies: Optional[Dict[str, str]] = None,
    token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    **kwargs,
):
    """
    Loads the tokenizer configuration from a pretrained model tokenizer configuration.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co.
            - a path to a *directory* containing a configuration file saved using the
              [`~PreTrainedTokenizer.save_pretrained`] method, e.g., `./my_model_directory/`.

        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download:
            Deprecated and ignored. All downloads are now resumed by default when possible.
            Will be removed in v5 of Transformers.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.

    <Tip>

    Passing `token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Dict`: The configuration of the tokenizer.

    Examples:

    ```python
    # Download configuration from huggingface.co and cache.
    tokenizer_config = get_tokenizer_config("google-bert/bert-base-uncased")
    # This model does not have a tokenizer config so the result will be an empty dict.
    tokenizer_config = get_tokenizer_config("FacebookAI/xlm-roberta-base")

    # Save a pretrained tokenizer locally and you can reload its config
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
    tokenizer.save_pretrained("tokenizer-test")
    tokenizer_config = get_tokenizer_config("tokenizer-test")
    ```"""
    use_auth_token = kwargs.pop("use_auth_token", None)
    if use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
            FutureWarning,
        )
        if token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        token = use_auth_token

    resolved_config_file = get_file_from_repo(
        pretrained_model_name_or_path,
        FEATURE_EXTRACTOR_NAME,
        cache_dir=cache_dir,
        force_download=force_download,
        resume_download=resume_download,
        proxies=proxies,
        token=token,
        revision=revision,
        local_files_only=local_files_only,
    )
    if resolved_config_file is None:
        logger.info(
            "Could not locate the feature extractor configuration file, will try to use the model config instead."
        )
        return {}

    with open(resolved_config_file, encoding="utf-8") as reader:
        return json.load(reader)


class AutoFeatureExtractor:
    r"""
    This is a generic feature extractor class that will be instantiated as one of the feature extractor classes of the
    library when created with the [`AutoFeatureExtractor.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoFeatureExtractor is designed to be instantiated "
            "using the `AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    @replace_list_option_in_docstrings(FEATURE_EXTRACTOR_MAPPING_NAMES)
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate one of the feature extractor classes of the library from a pretrained model vocabulary.

        The feature extractor class to instantiate is selected based on the `model_type` property of the config object
        (either passed as an argument or loaded from `pretrained_model_name_or_path` if possible), or when it's
        missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

        List options

        Params:
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
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final feature extractor object. If `True`, then this
                functions returns a `Tuple(feature_extractor, unused_kwargs)` where *unused_kwargs* is a dictionary
                consisting of the key/value pairs whose keys are not feature extractor attributes: i.e., the part of
                `kwargs` which has not been used to update `feature_extractor` and is otherwise ignored.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            kwargs (`Dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are feature extractor attributes will be used to override the
                loaded values. Behavior concerning key/value pairs whose keys are *not* feature extractor attributes is
                controlled by the `return_unused_kwargs` keyword parameter.

        <Tip>

        Passing `token=True` is required when you want to use a private model.

        </Tip>

        Examples:

        ```python
        >>> from transformers import AutoFeatureExtractor

        >>> # Download feature extractor from huggingface.co and cache.
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

        >>> # If feature extractor files are in a directory (e.g. feature extractor was saved using *save_pretrained('./test/saved_model/')*)
        >>> # feature_extractor = AutoFeatureExtractor.from_pretrained("./test/saved_model/")
        ```"""
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

        config = kwargs.pop("config", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        kwargs["_from_auto"] = True

        config_dict, _ = FeatureExtractionMixin.get_feature_extractor_dict(pretrained_model_name_or_path, **kwargs)
        feature_extractor_class = config_dict.get("feature_extractor_type", None)
        feature_extractor_auto_map = None
        if "AutoFeatureExtractor" in config_dict.get("auto_map", {}):
            feature_extractor_auto_map = config_dict["auto_map"]["AutoFeatureExtractor"]

        # If we don't find the feature extractor class in the feature extractor config, let's try the model config.
        if feature_extractor_class is None and feature_extractor_auto_map is None:
            if not isinstance(config, PretrainedConfig):
                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
                )
            # It could be in `config.feature_extractor_type``
            feature_extractor_class = getattr(config, "feature_extractor_type", None)
            if hasattr(config, "auto_map") and "AutoFeatureExtractor" in config.auto_map:
                feature_extractor_auto_map = config.auto_map["AutoFeatureExtractor"]

        if feature_extractor_class is not None:
            feature_extractor_class = feature_extractor_class_from_name(feature_extractor_class)

        has_remote_code = feature_extractor_auto_map is not None
        has_local_code = feature_extractor_class is not None or type(config) in FEATURE_EXTRACTOR_MAPPING
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
        )

        if has_remote_code and trust_remote_code:
            feature_extractor_class = get_class_from_dynamic_module(
                feature_extractor_auto_map, pretrained_model_name_or_path, **kwargs
            )
            _ = kwargs.pop("code_revision", None)
            if os.path.isdir(pretrained_model_name_or_path):
                feature_extractor_class.register_for_auto_class()
            return feature_extractor_class.from_dict(config_dict, **kwargs)
        elif feature_extractor_class is not None:
            return feature_extractor_class.from_dict(config_dict, **kwargs)
        # Last try: we use the FEATURE_EXTRACTOR_MAPPING.
        elif type(config) in FEATURE_EXTRACTOR_MAPPING:
            feature_extractor_class = FEATURE_EXTRACTOR_MAPPING[type(config)]
            return feature_extractor_class.from_dict(config_dict, **kwargs)

        raise ValueError(
            f"Unrecognized feature extractor in {pretrained_model_name_or_path}. Should have a "
            f"`feature_extractor_type` key in its {FEATURE_EXTRACTOR_NAME} of {CONFIG_NAME}, or one of the following "
            f"`model_type` keys in its {CONFIG_NAME}: {', '.join(c for c in FEATURE_EXTRACTOR_MAPPING_NAMES.keys())}"
        )

    @staticmethod
    def register(config_class, feature_extractor_class, exist_ok=False):
        """
        Register a new feature extractor for this class.

        Args:
            config_class ([`PretrainedConfig`]):
                The configuration corresponding to the model to register.
            feature_extractor_class ([`FeatureExtractorMixin`]): The feature extractor to register.
        """
        FEATURE_EXTRACTOR_MAPPING.register(config_class, feature_extractor_class, exist_ok=exist_ok)
