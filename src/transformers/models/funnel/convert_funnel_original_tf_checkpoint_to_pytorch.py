# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""Convert Funnel checkpoint."""

import argparse

import torch

from transformers import FunnelBaseModel, FunnelConfig, FunnelModel, load_tf_weights_in_funnel
from transformers.utils import logging


logging.set_verbosity_info()


def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path, base_model):
    # Initialise PyTorch model
    config = FunnelConfig.from_json_file(config_file)
    print(f"Building PyTorch model from configuration: {config}")
    model = FunnelBaseModel(config) if base_model else FunnelModel(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_funnel(model, config, tf_checkpoint_path)

    # Save pytorch-model
    print(f"Save PyTorch model to {pytorch_dump_path}")
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--tf_checkpoint_path", default=None, type=str, required=True, help="Path to the TensorFlow checkpoint path."
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained model. \nThis specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--base_model", action="store_true", help="Whether you want just the base model (no decoder) or not."
    )
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(
        args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path, args.base_model
    )


__all__ = []
