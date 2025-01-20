# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
"""Convert AIMv2 checkpoints from the original repository.

URL: https://github.com/apple/ml-aim/tree/main/aim-v2
"""

import argparse
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors import safe_open

from transformers import AIMv2Config, AIMv2Model, AutoImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_aimv2_config(model_name):
    config = AIMv2Config()

    # Use the appropriate hyperparameters depending on the model name.
    if "aimv2-base" in model_name:
        config.hidden_size = 768
        config.intermediate_size = 2048
        config.num_hidden_layers = 12
        config.num_attention_heads = 6
    elif "aimv2-large" in model_name:
        config.hidden_size = 1024
        config.intermediate_size = 2816
        config.num_hidden_layers = 24
        config.num_attention_heads = 8
    elif "aimv2-huge" in model_name:
        config.hidden_size = 1536
        config.intermediate_size = 4096
        config.num_hidden_layers = 24
        config.num_attention_heads = 12
    elif "aimv2-1B" in model_name:
        config.hidden_size = 2048
        config.intermediate_size = 5632
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
    elif "aimv2-3B" in model_name:
        config.hidden_size = 3072
        config.intermediate_size = 8192
        config.num_hidden_layers = 24
        config.num_attention_heads = 24

    return config


def create_rename_keys(config):
    rename_keys = []
    # fmt: off

    # patch embedding layer
    rename_keys.append(("preprocessor.pos_embed", "preprocessor.pos_embed"))
    rename_keys.append(("preprocessor.patchifier.proj.weight", "preprocessor.patchifier.proj.weight"))
    rename_keys.append(("preprocessor.patchifier.proj.bias", "preprocessor.patchifier.proj.bias"))
    rename_keys.append(("preprocessor.patchifier.norm.weight", "preprocessor.patchifier.norm.weight"))

    for i in range(config.num_hidden_layers):
        # attention blocks
        rename_keys.append((f"trunk.blocks.{i}.attn.qkv.weight", f"trunk.blocks.{i}.attn.qkv.weight"))
        rename_keys.append((f"trunk.blocks.{i}.attn.proj.weight", f"trunk.blocks.{i}.attn.proj.weight"))

        # MLP blocks
        rename_keys.append((f"trunk.blocks.{i}.norm_1.weight", f"trunk.blocks.{i}.norm_1.weight"))
        rename_keys.append((f"trunk.blocks.{i}.mlp.fc1.weight", f"trunk.blocks.{i}.mlp.fc1.weight"))
        rename_keys.append((f"trunk.blocks.{i}.mlp.fc2.weight", f"trunk.blocks.{i}.mlp.fc2.weight"))
        rename_keys.append((f"trunk.blocks.{i}.mlp.fc3.weight", f"trunk.blocks.{i}.mlp.fc3.weight"))
        rename_keys.append((f"trunk.blocks.{i}.norm_2.weight", f"trunk.blocks.{i}.norm_2.weight"))

    rename_keys.append(("trunk.post_trunk_norm.weight", "trunk.post_trunk_norm.weight"))

    # fmt: on
    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of a dog
def prepare_img():
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
    image = Image.open(requests.get(url, stream=True).raw)
    return image


@torch.no_grad()
def convert_aimv2_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our AIMv2 structure.
    """
    model_name_to_repo_id = {
        "aimv2-large-patch14-224": "apple/aimv2-large-patch14-224",
        "aimv2-huge-patch14-224": "apple/aimv2-huge-patch14-224",
        "aimv2-1B-patch14-224": "apple/aimv2-1B-patch14-224",
        "aimv2-3B-patch14-224": "apple/aimv2-3B-patch14-224",
        "aimv2-large-patch14-336": "apple/aimv2-large-patch14-336",
        "aimv2-huge-patch14-336": "apple/aimv2-huge-patch14-336",
        "aimv2-1B-patch14-336": "apple/aimv2-1B-patch14-336",
        "aimv2-3B-patch14-336": "apple/aimv2-3B-patch14-336",
        "aimv2-large-patch14-448": "apple/aimv2-large-patch14-448",
        "aimv2-huge-patch14-448": "apple/aimv2-huge-patch14-448",
        "aimv2-1B-patch14-448": "apple/aimv2-1B-patch14-448",
        "aimv2-3B-patch14-448": "apple/aimv2-3B-patch14-448",
        "aimv2-large-patch14-224-distilled": "apple/aimv2-large-patch14-224-distilled",
        "aimv2-large-patch14-336-distilled": "apple/aimv2-large-patch14-336-distilled",
        "aimv2-large-patch14-native": "apple/aimv2-large-patch14-native",
        "aimv2-large-patch14-224-lit": "apple/aimv2-large-patch14-224-lit",
    }

    # define default AIMv2 configuration
    config = get_aimv2_config(model_name)
    logger.info(f"Model config: {config}")

    # load original model from torch hub
    repo_id = model_name_to_repo_id[model_name]
    filename = "model.safetensors"

    filepath = hf_hub_download(repo_id=repo_id, filename=filename)
    state_dict = {}
    with safe_open(filepath, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    # load HuggingFace model
    model = AIMv2Model(config).eval()

    # rename keys
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)

    # assert values
    assert missing_keys == [], str(missing_keys)
    assert unexpected_keys == [], str(unexpected_keys)

    # load image
    image = prepare_img()
    # preprocess image
    preprocessor = AutoImageProcessor.from_pretrained("apple/aimv2-large-patch14-224")
    inputs = preprocessor(images=image, return_tensors="pt")

    with torch.no_grad():
        model(**inputs)

    print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        print(f"Saving preprocessor to {pytorch_dump_folder_path}")
        preprocessor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        model.push_to_hub(f"apple/{model_name}")
        preprocessor.push_to_hub(f"apple/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="aimv2-large-patch14-224",
        type=str,
        choices=[
            "aimv2-large-patch14-224",
            "aimv2-huge-patch14-224",
            "aimv2-1B-patch14-224",
            "aimv2-3B-patch14-224",
            "aimv2-large-patch14-336",
            "aimv2-huge-patch14-336",
            "aimv2-1B-patch14-336",
            "aimv2-3B-patch14-336",
            "aimv2-large-patch14-448",
            "aimv2-huge-patch14-448",
            "aimv2-1B-patch14-448",
            "aimv2-3B-patch14-448",
            "aimv2-large-patch14-224-distilled",
            "aimv2-large-patch14-336-distilled",
            "aimv2-large-patch14-native",
            "aimv2-large-patch14-224-lit",
        ],
        help="Name of the AIMv2 model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the converted model to the 🤗 hub.",
    )

    args = parser.parse_args()
    convert_aimv2_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
