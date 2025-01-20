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
"""AutoProcessor class."""

import importlib
import inspect
import json
import os
import warnings
from collections import OrderedDict

# Build the list of all feature extractors
from ...configuration_utils import PretrainedConfig
from ...dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from ...feature_extraction_utils import FeatureExtractionMixin
from ...image_processing_utils import ImageProcessingMixin
from ...processing_utils import ProcessorMixin
from ...tokenization_utils import TOKENIZER_CONFIG_FILE
from ...utils import FEATURE_EXTRACTOR_NAME, PROCESSOR_NAME, get_file_from_repo, logging
from .auto_factory import _LazyAutoMapping
from .configuration_auto import (
    CONFIG_MAPPING_NAMES,
    AutoConfig,
    model_type_to_module_name,
    replace_list_option_in_docstrings,
)
from .feature_extraction_auto import AutoFeatureExtractor
from .image_processing_auto import AutoImageProcessor
from .tokenization_auto import AutoTokenizer


logger = logging.get_logger(__name__)

PROCESSOR_MAPPING_NAMES = OrderedDict(
    [
        ("align", "AlignProcessor"),
        ("altclip", "AltCLIPProcessor"),
        ("aria", "AriaProcessor"),
        ("bark", "BarkProcessor"),
        ("blip", "BlipProcessor"),
        ("blip-2", "Blip2Processor"),
        ("bridgetower", "BridgeTowerProcessor"),
        ("chameleon", "ChameleonProcessor"),
        ("chinese_clip", "ChineseCLIPProcessor"),
        ("clap", "ClapProcessor"),
        ("clip", "CLIPProcessor"),
        ("clipseg", "CLIPSegProcessor"),
        ("clvp", "ClvpProcessor"),
        ("colpali", "ColPaliProcessor"),
        ("emu3", "Emu3Processor"),
        ("flava", "FlavaProcessor"),
        ("fuyu", "FuyuProcessor"),
        ("git", "GitProcessor"),
        ("grounding-dino", "GroundingDinoProcessor"),
        ("groupvit", "CLIPProcessor"),
        ("hubert", "Wav2Vec2Processor"),
        ("idefics", "IdeficsProcessor"),
        ("idefics2", "Idefics2Processor"),
        ("idefics3", "Idefics3Processor"),
        ("instructblip", "InstructBlipProcessor"),
        ("instructblipvideo", "InstructBlipVideoProcessor"),
        ("kosmos-2", "Kosmos2Processor"),
        ("layoutlmv2", "LayoutLMv2Processor"),
        ("layoutlmv3", "LayoutLMv3Processor"),
        ("llava", "LlavaProcessor"),
        ("llava_next", "LlavaNextProcessor"),
        ("llava_next_video", "LlavaNextVideoProcessor"),
        ("llava_onevision", "LlavaOnevisionProcessor"),
        ("markuplm", "MarkupLMProcessor"),
        ("mctct", "MCTCTProcessor"),
        ("mgp-str", "MgpstrProcessor"),
        ("mllama", "MllamaProcessor"),
        ("moonshine", "Wav2Vec2Processor"),
        ("oneformer", "OneFormerProcessor"),
        ("owlv2", "Owlv2Processor"),
        ("owlvit", "OwlViTProcessor"),
        ("paligemma", "PaliGemmaProcessor"),
        ("pix2struct", "Pix2StructProcessor"),
        ("pixtral", "PixtralProcessor"),
        ("pop2piano", "Pop2PianoProcessor"),
        ("qwen2_5_vl", "Qwen2_5_VLProcessor"),
        ("qwen2_audio", "Qwen2AudioProcessor"),
        ("qwen2_vl", "Qwen2VLProcessor"),
        ("sam", "SamProcessor"),
        ("seamless_m4t", "SeamlessM4TProcessor"),
        ("sew", "Wav2Vec2Processor"),
        ("sew-d", "Wav2Vec2Processor"),
        ("siglip", "SiglipProcessor"),
        ("speech_to_text", "Speech2TextProcessor"),
        ("speech_to_text_2", "Speech2Text2Processor"),
        ("speecht5", "SpeechT5Processor"),
        ("trocr", "TrOCRProcessor"),
        ("tvlt", "TvltProcessor"),
        ("tvp", "TvpProcessor"),
        ("udop", "UdopProcessor"),
        ("unispeech", "Wav2Vec2Processor"),
        ("unispeech-sat", "Wav2Vec2Processor"),
        ("video_llava", "VideoLlavaProcessor"),
        ("vilt", "ViltProcessor"),
        ("vipllava", "LlavaProcessor"),
        ("vision-text-dual-encoder", "VisionTextDualEncoderProcessor"),
        ("wav2vec2", "Wav2Vec2Processor"),
        ("wav2vec2-bert", "Wav2Vec2Processor"),
        ("wav2vec2-conformer", "Wav2Vec2Processor"),
        ("wavlm", "Wav2Vec2Processor"),
        ("whisper", "WhisperProcessor"),
        ("xclip", "XCLIPProcessor"),
    ]
)

PROCESSOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, PROCESSOR_MAPPING_NAMES)


def processor_class_from_name(class_name: str):
    for module_name, processors in PROCESSOR_MAPPING_NAMES.items():
        if class_name in processors:
            module_name = model_type_to_module_name(module_name)

            module = importlib.import_module(f".{module_name}", "transformers.models")
            try:
                return getattr(module, class_name)
            except AttributeError:
                continue

    for processor in PROCESSOR_MAPPING._extra_content.values():
        if getattr(processor, "__name__", None) == class_name:
            return processor

    # We did not fine the class, but maybe it's because a dep is missing. In that case, the class will be in the main
    # init and we return the proper dummy to get an appropriate error message.
    main_module = importlib.import_module("transformers")
    if hasattr(main_module, class_name):
        return getattr(main_module, class_name)

    return None


class AutoProcessor:
    r"""
    This is a generic processor class that will be instantiated as one of the processor classes of the library when
    created with the [`AutoProcessor.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoProcessor is designed to be instantiated "
            "using the `AutoProcessor.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    @replace_list_option_in_docstrings(PROCESSOR_MAPPING_NAMES)
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate one of the processor classes of the library from a pretrained model vocabulary.

        The processor class to instantiate is selected based on the `model_type` property of the config object (either
        passed as an argument or loaded from `pretrained_model_name_or_path` if possible):

        List options

        Params:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained feature_extractor hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a processor files saved using the `save_pretrained()` method,
                  e.g., `./my_model_directory/`.
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
        >>> from transformers import AutoProcessor

        >>> # Download processor from huggingface.co and cache.
        >>> processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")

        >>> # If processor files are in a directory (e.g. processor was saved using *save_pretrained('./test/saved_model/')*)
        >>> # processor = AutoProcessor.from_pretrained("./test/saved_model/")
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

        processor_class = None
        processor_auto_map = None

        # First, let's see if we have a processor or preprocessor config.
        # Filter the kwargs for `get_file_from_repo`.
        get_file_from_repo_kwargs = {
            key: kwargs[key] for key in inspect.signature(get_file_from_repo).parameters.keys() if key in kwargs
        }

        # Let's start by checking whether the processor class is saved in a processor config
        processor_config_file = get_file_from_repo(
            pretrained_model_name_or_path, PROCESSOR_NAME, **get_file_from_repo_kwargs
        )
        if processor_config_file is not None:
            config_dict, _ = ProcessorMixin.get_processor_dict(pretrained_model_name_or_path, **kwargs)
            processor_class = config_dict.get("processor_class", None)
            if "AutoProcessor" in config_dict.get("auto_map", {}):
                processor_auto_map = config_dict["auto_map"]["AutoProcessor"]

        if processor_class is None:
            # If not found, let's check whether the processor class is saved in an image processor config
            preprocessor_config_file = get_file_from_repo(
                pretrained_model_name_or_path, FEATURE_EXTRACTOR_NAME, **get_file_from_repo_kwargs
            )
            if preprocessor_config_file is not None:
                config_dict, _ = ImageProcessingMixin.get_image_processor_dict(pretrained_model_name_or_path, **kwargs)
                processor_class = config_dict.get("processor_class", None)
                if "AutoProcessor" in config_dict.get("auto_map", {}):
                    processor_auto_map = config_dict["auto_map"]["AutoProcessor"]

            # If not found, let's check whether the processor class is saved in a feature extractor config
            if preprocessor_config_file is not None and processor_class is None:
                config_dict, _ = FeatureExtractionMixin.get_feature_extractor_dict(
                    pretrained_model_name_or_path, **kwargs
                )
                processor_class = config_dict.get("processor_class", None)
                if "AutoProcessor" in config_dict.get("auto_map", {}):
                    processor_auto_map = config_dict["auto_map"]["AutoProcessor"]

        if processor_class is None:
            # Next, let's check whether the processor class is saved in a tokenizer
            tokenizer_config_file = get_file_from_repo(
                pretrained_model_name_or_path, TOKENIZER_CONFIG_FILE, **get_file_from_repo_kwargs
            )
            if tokenizer_config_file is not None:
                with open(tokenizer_config_file, encoding="utf-8") as reader:
                    config_dict = json.load(reader)

                processor_class = config_dict.get("processor_class", None)
                if "AutoProcessor" in config_dict.get("auto_map", {}):
                    processor_auto_map = config_dict["auto_map"]["AutoProcessor"]

        if processor_class is None:
            # Otherwise, load config, if it can be loaded.
            if not isinstance(config, PretrainedConfig):
                config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
                )

            # And check if the config contains the processor class.
            processor_class = getattr(config, "processor_class", None)
            if hasattr(config, "auto_map") and "AutoProcessor" in config.auto_map:
                processor_auto_map = config.auto_map["AutoProcessor"]

        if processor_class is not None:
            processor_class = processor_class_from_name(processor_class)

        has_remote_code = processor_auto_map is not None
        has_local_code = processor_class is not None or type(config) in PROCESSOR_MAPPING
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
        )

        if has_remote_code and trust_remote_code:
            processor_class = get_class_from_dynamic_module(
                processor_auto_map, pretrained_model_name_or_path, **kwargs
            )
            _ = kwargs.pop("code_revision", None)
            if os.path.isdir(pretrained_model_name_or_path):
                processor_class.register_for_auto_class()
            return processor_class.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
            )
        elif processor_class is not None:
            return processor_class.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
            )
        # Last try: we use the PROCESSOR_MAPPING.
        elif type(config) in PROCESSOR_MAPPING:
            return PROCESSOR_MAPPING[type(config)].from_pretrained(pretrained_model_name_or_path, **kwargs)

        # At this stage, there doesn't seem to be a `Processor` class available for this model, so let's try a
        # tokenizer.
        try:
            return AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
            )
        except Exception:
            try:
                return AutoImageProcessor.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
                )
            except Exception:
                pass

            try:
                return AutoFeatureExtractor.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
                )
            except Exception:
                pass

        raise ValueError(
            f"Unrecognized processing class in {pretrained_model_name_or_path}. Can't instantiate a processor, a "
            "tokenizer, an image processor or a feature extractor for this model. Make sure the repository contains "
            "the files of at least one of those processing classes."
        )

    @staticmethod
    def register(config_class, processor_class, exist_ok=False):
        """
        Register a new processor for this class.

        Args:
            config_class ([`PretrainedConfig`]):
                The configuration corresponding to the model to register.
            processor_class ([`FeatureExtractorMixin`]): The processor to register.
        """
        PROCESSOR_MAPPING.register(config_class, processor_class, exist_ok=exist_ok)
