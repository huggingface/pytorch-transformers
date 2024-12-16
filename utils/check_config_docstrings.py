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

import inspect
import re

from transformers.utils import direct_transformers_import


# All paths are set with the intent you should run this script from the root of the repo with the command
# python utils/check_config_docstrings.py
PATH_TO_TRANSFORMERS = "src/transformers"


# This is to make sure the transformers module imported is the one in the repo.
transformers = direct_transformers_import(PATH_TO_TRANSFORMERS)

CONFIG_MAPPING = transformers.models.auto.configuration_auto.CONFIG_MAPPING

# Regex pattern used to find the checkpoint mentioned in the docstring of `config_class`.
# For example, `[google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased)`
_re_checkpoint = re.compile(r"\[(.+?)\]\((https://huggingface\.co/.+?)\)")


CONFIG_CLASSES_TO_IGNORE_FOR_DOCSTRING_CHECKPOINT_CHECK = {
    "DecisionTransformerConfig",
    "EncoderDecoderConfig",
    "MusicgenConfig",
    "RagConfig",
    "SpeechEncoderDecoderConfig",
    "TimmBackboneConfig",
    "TimmWrapperConfig",
    "VisionEncoderDecoderConfig",
    "VisionTextDualEncoderConfig",
    "LlamaConfig",
    "GraniteConfig",
    "GraniteMoeConfig",
}


def get_checkpoint_from_config_class(config_class):
    checkpoint = None

    # source code of `config_class`
    config_source = inspect.getsource(config_class)
    checkpoints = _re_checkpoint.findall(config_source)

    # Each `checkpoint` is a tuple of a checkpoint name and a checkpoint link.
    # For example, `('google-bert/bert-base-uncased', 'https://huggingface.co/google-bert/bert-base-uncased')`
    for ckpt_name, ckpt_link in checkpoints:
        # allow the link to end with `/`
        if ckpt_link.endswith("/"):
            ckpt_link = ckpt_link[:-1]

        # verify the checkpoint name corresponds to the checkpoint link
        ckpt_link_from_name = f"https://huggingface.co/{ckpt_name}"
        if ckpt_link == ckpt_link_from_name:
            checkpoint = ckpt_name
            break

    return checkpoint


def check_config_docstrings_have_checkpoints():
    configs_without_checkpoint = []

    for config_class in list(CONFIG_MAPPING.values()):
        # Skip deprecated models
        if "models.deprecated" in config_class.__module__:
            continue
        checkpoint = get_checkpoint_from_config_class(config_class)

        name = config_class.__name__
        if checkpoint is None and name not in CONFIG_CLASSES_TO_IGNORE_FOR_DOCSTRING_CHECKPOINT_CHECK:
            configs_without_checkpoint.append(name)

    if len(configs_without_checkpoint) > 0:
        message = "\n".join(sorted(configs_without_checkpoint))
        raise ValueError(
            f"The following configurations don't contain any valid checkpoint:\n{message}\n\n"
            "The requirement is to include a link pointing to one of the models of this architecture in the "
            "docstring of the config classes listed above. The link should have be a markdown format like "
            "[myorg/mymodel](https://huggingface.co/myorg/mymodel)."
        )


if __name__ == "__main__":
    check_config_docstrings_have_checkpoints()
