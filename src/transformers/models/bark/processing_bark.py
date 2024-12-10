# coding=utf-8
# Copyright 2023 The Suno AI Authors and The HuggingFace Inc. team. All rights reserved.
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
Processor class for Bark
"""

import json
import os
from typing import Optional

import numpy as np

from ...feature_extraction_utils import BatchFeature
from ...processing_utils import ProcessorMixin
from ...utils import logging
from ...utils.hub import get_file_from_repo
from ..auto import AutoTokenizer


logger = logging.get_logger(__name__)


class BarkProcessor(ProcessorMixin):
    r"""
    Constructs a Bark processor which wraps a text tokenizer and optional Bark voice presets into a single processor.

    Args:
        tokenizer ([`PreTrainedTokenizer`]):
            An instance of [`PreTrainedTokenizer`].
        speaker_embeddings (`Dict[Dict[str]]`, *optional*):
            Optional nested speaker embeddings dictionary. The first level contains voice preset names (e.g
            `"en_speaker_4"`). The second level contains `"semantic_prompt"`, `"coarse_prompt"` and `"fine_prompt"`
            embeddings. The values correspond to the path of the corresponding `np.ndarray`. See
            [here](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c) for
            a list of `voice_preset_names`.

    """

    tokenizer_class = "AutoTokenizer"
    attributes = ["tokenizer"]

    preset_shape = {
        "semantic_prompt": 1,
        "coarse_prompt": 2,
        "fine_prompt": 2,
    }

    def __init__(self, tokenizer, speaker_embeddings=None):
        super().__init__(tokenizer)

        self.speaker_embeddings = speaker_embeddings

    @classmethod
    def from_pretrained(
        cls, pretrained_processor_name_or_path, speaker_embeddings_dict_path="speaker_embeddings_path.json", **kwargs
    ):
        r"""
        Instantiate a Bark processor associated with a pretrained model.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained [`BarkProcessor`] hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a processor saved using the [`~BarkProcessor.save_pretrained`]
                  method, e.g., `./my_model_directory/`.
            speaker_embeddings_dict_path (`str`, *optional*, defaults to `"speaker_embeddings_path.json"`):
                The name of the `.json` file containing the speaker_embeddings dictionnary located in
                `pretrained_model_name_or_path`. If `None`, no speaker_embeddings is loaded.
            **kwargs
                Additional keyword arguments passed along to both
                [`~tokenization_utils_base.PreTrainedTokenizer.from_pretrained`].
        """

        if speaker_embeddings_dict_path is not None:
            speaker_embeddings_path = get_file_from_repo(
                pretrained_processor_name_or_path,
                speaker_embeddings_dict_path,
                subfolder=kwargs.pop("subfolder", None),
                cache_dir=kwargs.pop("cache_dir", None),
                force_download=kwargs.pop("force_download", False),
                proxies=kwargs.pop("proxies", None),
                resume_download=kwargs.pop("resume_download", None),
                local_files_only=kwargs.pop("local_files_only", False),
                token=kwargs.pop("use_auth_token", None),
                revision=kwargs.pop("revision", None),
            )
            if speaker_embeddings_path is None:
                logger.warning(
                    f"""`{os.path.join(pretrained_processor_name_or_path,speaker_embeddings_dict_path)}` does not exists
                    , no preloaded speaker embeddings will be used - Make sure to provide a correct path to the json
                    dictionnary if wanted, otherwise set `speaker_embeddings_dict_path=None`."""
                )
                speaker_embeddings = None
            else:
                with open(speaker_embeddings_path) as speaker_embeddings_json:
                    speaker_embeddings = json.load(speaker_embeddings_json)
        else:
            speaker_embeddings = None

        tokenizer = AutoTokenizer.from_pretrained(pretrained_processor_name_or_path, **kwargs)

        return cls(tokenizer=tokenizer, speaker_embeddings=speaker_embeddings)

    def save_pretrained(
        self,
        save_directory,
        speaker_embeddings_dict_path="speaker_embeddings_path.json",
        speaker_embeddings_directory="speaker_embeddings",
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Saves the attributes of this processor (tokenizer...) in the specified directory so that it can be reloaded
        using the [`~BarkProcessor.from_pretrained`] method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the tokenizer files and the speaker embeddings will be saved (directory will be created
                if it does not exist).
            speaker_embeddings_dict_path (`str`, *optional*, defaults to `"speaker_embeddings_path.json"`):
                The name of the `.json` file that will contains the speaker_embeddings nested path dictionnary, if it
                exists, and that will be located in `pretrained_model_name_or_path/speaker_embeddings_directory`.
            speaker_embeddings_directory (`str`, *optional*, defaults to `"speaker_embeddings/"`):
                The name of the folder in which the speaker_embeddings arrays will be saved.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs:
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        if self.speaker_embeddings is not None:
            os.makedirs(os.path.join(save_directory, speaker_embeddings_directory, "v2"), exist_ok=True)

            embeddings_dict = {}

            embeddings_dict["repo_or_path"] = save_directory

            for prompt_key in self.speaker_embeddings:
                if prompt_key != "repo_or_path":
                    voice_preset = self._load_voice_preset(prompt_key)

                    tmp_dict = {}
                    for key in self.speaker_embeddings[prompt_key]:
                        np.save(
                            os.path.join(
                                embeddings_dict["repo_or_path"], speaker_embeddings_directory, f"{prompt_key}_{key}"
                            ),
                            voice_preset[key],
                            allow_pickle=False,
                        )
                        tmp_dict[key] = os.path.join(speaker_embeddings_directory, f"{prompt_key}_{key}.npy")

                    embeddings_dict[prompt_key] = tmp_dict

            with open(os.path.join(save_directory, speaker_embeddings_dict_path), "w") as fp:
                json.dump(embeddings_dict, fp)

        super().save_pretrained(save_directory, push_to_hub, **kwargs)

    def _load_voice_preset(self, voice_preset: str = None, **kwargs):
        voice_preset_paths = self.speaker_embeddings[voice_preset]

        voice_preset_dict = {}
        for key in ["semantic_prompt", "coarse_prompt", "fine_prompt"]:
            if key not in voice_preset_paths:
                raise ValueError(
                    f"Voice preset unrecognized, missing {key} as a key in self.speaker_embeddings[{voice_preset}]."
                )

            path = get_file_from_repo(
                self.speaker_embeddings.get("repo_or_path", "/"),
                voice_preset_paths[key],
                subfolder=kwargs.pop("subfolder", None),
                cache_dir=kwargs.pop("cache_dir", None),
                force_download=kwargs.pop("force_download", False),
                proxies=kwargs.pop("proxies", None),
                resume_download=kwargs.pop("resume_download", None),
                local_files_only=kwargs.pop("local_files_only", False),
                token=kwargs.pop("use_auth_token", None),
                revision=kwargs.pop("revision", None),
            )
            if path is None:
                raise ValueError(
                    f"""`{os.path.join(self.speaker_embeddings.get("repo_or_path", "/"),voice_preset_paths[key])}` does not exists
                    , no preloaded voice preset will be used - Make sure to provide correct paths to the {voice_preset}
                    embeddings."""
                )

            voice_preset_dict[key] = np.load(path)

        return voice_preset_dict

    def _validate_voice_preset_dict(self, voice_preset: Optional[dict] = None):
        for key in ["semantic_prompt", "coarse_prompt", "fine_prompt"]:
            if key not in voice_preset:
                raise ValueError(f"Voice preset unrecognized, missing {key} as a key.")

            if not isinstance(voice_preset[key], np.ndarray):
                raise TypeError(f"{key} voice preset must be a {str(self.preset_shape[key])}D ndarray.")

            if len(voice_preset[key].shape) != self.preset_shape[key]:
                raise ValueError(f"{key} voice preset must be a {str(self.preset_shape[key])}D ndarray.")

    def __call__(
        self,
        text=None,
        voice_preset=None,
        return_tensors="pt",
        max_length=256,
        add_special_tokens=False,
        return_attention_mask=True,
        return_token_type_ids=False,
        **kwargs,
    ):
        """
        Main method to prepare for the model one or several sequences(s). This method forwards the `text` and `kwargs`
        arguments to the AutoTokenizer's [`~AutoTokenizer.__call__`] to encode the text. The method also proposes a
        voice preset which is a dictionary of arrays that conditions `Bark`'s output. `kwargs` arguments are forwarded
        to the tokenizer and to `cached_file` method if `voice_preset` is a valid filename.

        Args:
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            voice_preset (`str`, `Dict[np.ndarray]`):
                The voice preset, i.e the speaker embeddings. It can either be a valid voice_preset name, e.g
                `"en_speaker_1"`, or directly a dictionnary of `np.ndarray` embeddings for each submodel of `Bark`. Or
                it can be a valid file name of a local `.npz` single voice preset.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:

                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.

        Returns:
            Tuple([`BatchEncoding`], [`BatchFeature`]): A tuple composed of a [`BatchEncoding`], i.e the output of the
            `tokenizer` and a [`BatchFeature`], i.e the voice preset with the right tensors type.
        """
        if voice_preset is not None and not isinstance(voice_preset, dict):
            if (
                isinstance(voice_preset, str)
                and self.speaker_embeddings is not None
                and voice_preset in self.speaker_embeddings
            ):
                voice_preset = self._load_voice_preset(voice_preset)

            else:
                if isinstance(voice_preset, str) and not voice_preset.endswith(".npz"):
                    voice_preset = voice_preset + ".npz"

                voice_preset = np.load(voice_preset)

        if voice_preset is not None:
            self._validate_voice_preset_dict(voice_preset, **kwargs)
            voice_preset = BatchFeature(data=voice_preset, tensor_type=return_tensors)

        encoded_text = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding="max_length",
            max_length=max_length,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )

        if voice_preset is not None:
            encoded_text["history_prompt"] = voice_preset

        return encoded_text


__all__ = ["BarkProcessor"]
