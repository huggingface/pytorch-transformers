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
from argparse import ArgumentParser, Namespace
import os

from ..utils import logging
from . import BaseTransformersCLICommand


def convert_command_factory(args: Namespace):
    """
    Factory function used to convert a model TF 1.0 checkpoint in a PyTorch checkpoint.

    Returns: ServeCommand
    """
    return ConvertCommand(
        args.model_type, args.tf_checkpoint, args.pytorch_dump_output, args.config, args.finetuning_task_name
    )


IMPORT_ERROR_MESSAGE = """
transformers can only be used from the commandline to convert TensorFlow models in PyTorch, In that case, it requires
TensorFlow to be installed. Please see https://www.tensorflow.org/install/ for installation instructions.
"""


class ConvertCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        """
        Register this command to argparse so it's available for the transformer-cli

        Args:
            parser: Root parser to register command-specific arguments
        """
        train_parser = parser.add_parser(
            "convert",
            help="CLI tool to run convert model from original author checkpoints to Transformers PyTorch checkpoints.",
        )
        train_parser.add_argument("--model_type", type=str, required=True, help="Model's type.")
        train_parser.add_argument(
            "--tf_checkpoint", type=str, required=True, help="TensorFlow checkpoint path or folder."
        )
        train_parser.add_argument(
            "--pytorch_dump_output", type=str, required=True, help="Path to the PyTorch saved model output."
        )
        train_parser.add_argument("--config", type=str, default="", help="Configuration file path or folder.")
        train_parser.add_argument(
            "--finetuning_task_name",
            type=str,
            default=None,
            help="Optional fine-tuning task name if the TF model was a finetuned model.",
        )
        train_parser.set_defaults(func=convert_command_factory)

    def __init__(
        self,
        model_type: str,
        tf_checkpoint: str,
        pytorch_dump_output: str,
        config: str,
        finetuning_task_name: str,
        *args,
    ):
        self._logger = logging.get_logger("transformers-cli/converting")

        self._logger.info(f"Loading model {model_type}")
        self._model_type = model_type
        self._tf_checkpoint = tf_checkpoint
        self._pytorch_dump_output = pytorch_dump_output
        self._config = config
        self._finetuning_task_name = finetuning_task_name

    def run(self):
        # Check if checkpoint path exists
        if not os.path.exists(self._tf_checkpoint):
            raise FileNotFoundError(f"TensorFlow checkpoint path {self._tf_checkpoint} does not exist.")
        
        # Check if output path exists
        output_dir = os.path.dirname(self._pytorch_dump_output)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if self._model_type == "albert":
            try:
                from ..models.albert.convert_albert_original_tf_checkpoint_to_pytorch import (
                    convert_tf_checkpoint_to_pytorch,
                )
            except ImportError:
                raise ImportError(IMPORT_ERROR_MESSAGE)

            convert_tf_checkpoint_to_pytorch(self._tf_checkpoint, self._config, self._pytorch_dump_output)
        elif self._model_type == "bert":
            try:
                from ..models.bert.convert_bert_original_tf_checkpoint_to_pytorch import (
                    convert_tf_checkpoint_to_pytorch,
                )
            except ImportError:
                raise ImportError(IMPORT_ERROR_MESSAGE)

            convert_tf_checkpoint_to_pytorch(self._tf_checkpoint, self._config, self._pytorch_dump_output)
        elif self._model_type == "funnel":
            try:
                from ..models.funnel.convert_funnel_original_tf_checkpoint_to_pytorch import (
                    convert_tf_checkpoint_to_pytorch,
                )
            except ImportError:
                raise ImportError(IMPORT_ERROR_MESSAGE)

            convert_tf_checkpoint_to_pytorch(self._tf_checkpoint, self._config, self._pytorch_dump_output)
        elif self._model_type == "t5":
            try:
                from ..models.t5.convert_t5_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch
            except ImportError:
                raise ImportError(IMPORT_ERROR_MESSAGE)

            convert_tf_checkpoint_to_pytorch(self._tf_checkpoint, self._config, self._pytorch_dump_output)
        elif self._model_type == "gpt":
            from ..models.openai.convert_openai_original_tf_checkpoint_to_pytorch import (
                convert_openai_checkpoint_to_pytorch,
            )

            convert_openai_checkpoint_to_pytorch(self._tf_checkpoint, self._config, self._pytorch_dump_output)
        elif self._model_type == "gpt2":
            try:
                from ..models.gpt2.convert_gpt2_original_tf_checkpoint_to_pytorch import (
                    convert_gpt2_checkpoint_to_pytorch,
                )
            except ImportError:
                raise ImportError(IMPORT_ERROR_MESSAGE)

            convert_gpt2_checkpoint_to_pytorch(self._tf_checkpoint, self._config, self._pytorch_dump_output)
        elif self._model_type == "xlnet":
            try:
                from ..models.xlnet.convert_xlnet_original_tf_checkpoint_to_pytorch import (
                    convert_xlnet_checkpoint_to_pytorch,
                )
            except ImportError:
                raise ImportError(IMPORT_ERROR_MESSAGE)

            convert_xlnet_checkpoint_to_pytorch(
                self._tf_checkpoint, self._config, self._pytorch_dump_output, self._finetuning_task_name
            )
        elif self._model_type == "xlm":
            from ..models.xlm.convert_xlm_original_pytorch_checkpoint_to_pytorch import (
                convert_xlm_checkpoint_to_pytorch,
            )

            convert_xlm_checkpoint_to_pytorch(self._tf_checkpoint, self._pytorch_dump_output)
        elif self._model_type == "lxmert":
            from ..models.lxmert.convert_lxmert_original_tf_checkpoint_to_pytorch import (
                convert_lxmert_checkpoint_to_pytorch,
            )

            convert_lxmert_checkpoint_to_pytorch(self._tf_checkpoint, self._pytorch_dump_output)
        elif self._model_type == "rembert":
            from ..models.rembert.convert_rembert_tf_checkpoint_to_pytorch import (
                convert_rembert_tf_checkpoint_to_pytorch,
            )

            convert_rembert_tf_checkpoint_to_pytorch(self._tf_checkpoint, self._config, self._pytorch_dump_output)
        else:
            raise ValueError(f"--model_type should be selected from the list {list(conversion_functions.keys())}")
        self._logger.info(f"Conversion completed successfully. Model saved to {self._pytorch_dump_output}.")
        except ImportError:
            raise ImportError(IMPORT_ERROR_MESSAGE)
