# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from .generation.configuration_utils import GenerationConfig
from .training_args import TrainingArguments
from .utils import add_start_docstrings


logger = logging.getLogger(__name__)


@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class Seq2SeqTrainingArguments(TrainingArguments):
    """
    Args:
        predict_with_generate (`bool`, *optional*, defaults to `False`):
            Whether to use generate to calculate generative metrics (ROUGE, BLEU).
        generation_max_length (`int`, *optional*):
            The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default to the
            `max_length` value of the model configuration.
        generation_num_beams (`int`, *optional*):
            The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default to the
            `num_beams` value of the model configuration.
        generation_config (`str` or `Path` or [`~generation.GenerationConfig`], *optional*):
            Allows to load a [`~generation.GenerationConfig`] from the `from_pretrained` method. This can be either:

            - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
              huggingface.co.
            - a path to a *directory* containing a configuration file saved using the
              [`~GenerationConfig.save_pretrained`] method, e.g., `./my_model_directory/`.
            - a [`~generation.GenerationConfig`] object.
    """

    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to use SortishSampler or not."})
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    generation_max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                "to the `max_length` value of the model configuration."
            )
        },
    )
    generation_num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                "to the `num_beams` value of the model configuration."
            )
        },
    )
    generation_config: Optional[Union[str, Path, GenerationConfig]] = field(
        default=None,
        metadata={
            "help": "Model id, file path or url pointing to a GenerationConfig json file, to use during prediction."
        },
    )

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values and `GenerationConfig` by dictionaries (for JSON
        serialization support). It obfuscates the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = super().to_dict()
        for k, v in d.items():
            if isinstance(v, GenerationConfig):
                d[k] = v.to_dict()
        return d
