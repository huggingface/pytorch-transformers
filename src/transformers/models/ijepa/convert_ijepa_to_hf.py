# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
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
"""Convert IJEPA checkpoints from the original repository.

URL: https://github.com/facebookresearch/ijepa
"""

import argparse
import gc
import re
from pathlib import Path

import requests
import torch
from PIL import Image

from transformers import (
    IJepaConfig,
    IJepaModel,
    ViTImageProcessor,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # Projection layer + position embeddings
    r"pos_embed":                               r"embeddings.position_embeddings",
    r"patch_embed.proj.weight":                 r"embeddings.patch_embeddings.projection.weight",
    r"patch_embed.proj.bias":                   r"embeddings.patch_embeddings.projection.bias",

    # Encoder layers: Layernorms, Attention, Feedforward layers
    r"blocks.(\d+).norm1.weight":               r"encoder.layer.\1.layernorm_before.weight",
    r"blocks.(\d+).norm1.bias":                 r"encoder.layer.\1.layernorm_before.bias",
    r"blocks.(\d+).attn.proj.weight":           r"encoder.layer.\1.attention.output.dense.weight",
    r"blocks.(\d+).attn.proj.bias":             r"encoder.layer.\1.attention.output.dense.bias",
    r"blocks.(\d+).norm2.weight":               r"encoder.layer.\1.layernorm_after.weight",
    r"blocks.(\d+).norm2.bias":                 r"encoder.layer.\1.layernorm_after.bias",
    r"blocks.(\d+).mlp.fc1.weight":             r"encoder.layer.\1.intermediate.dense.weight",
    r"blocks.(\d+).mlp.fc1.bias":               r"encoder.layer.\1.intermediate.dense.bias",
    r"blocks.(\d+).mlp.fc2.weight":             r"encoder.layer.\1.output.dense.weight",
    r"blocks.(\d+).mlp.fc2.bias":               r"encoder.layer.\1.output.dense.bias",

    # Layernorm + pooler
    r"norm.weight":                             r"layernorm.weight",
    r"norm.bias":                               r"layernorm.bias",
}
# fmt: on


def convert_old_keys_to_new_keys(state_dict_keys: dict = None):
    """
    Converts old keys to new keys using the mapping and dynamically removes the 'ijepa.' prefix if necessary.

    Args:
        state_dict_keys (dict): The keys from the state_dict to convert.

    Returns:
        dict: A mapping from old keys to new keys.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text

        # Apply regex-based mapping
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)  # Skip the key
                continue
            new_text = re.sub(pattern, replacement, new_text)

        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))

    return output_dict


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config):
    for i in range(config.num_hidden_layers):
        # read in weights + bias of input projection layer (in timm, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"blocks.{i}.attn.qkv.weight")
        in_proj_bias = state_dict.pop(f"blocks.{i}.attn.qkv.bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"encoder.layer.{i}.attention.attention.query.weight"] = in_proj_weight[: config.hidden_size, :]
        state_dict[f"encoder.layer.{i}.attention.attention.query.bias"] = in_proj_bias[: config.hidden_size]
        state_dict[f"encoder.layer.{i}.attention.attention.key.weight"] = in_proj_weight[
            config.hidden_size : config.hidden_size * 2, :
        ]
        state_dict[f"encoder.layer.{i}.attention.attention.key.bias"] = in_proj_bias[
            config.hidden_size : config.hidden_size * 2
        ]
        state_dict[f"encoder.layer.{i}.attention.attention.value.weight"] = in_proj_weight[-config.hidden_size :, :]
        state_dict[f"encoder.layer.{i}.attention.attention.value.bias"] = in_proj_bias[-config.hidden_size :]


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


def get_ijepa_config(model_name):
    patch_size = int(model_name.split("_")[1][4:])
    config = IJepaConfig(patch_size=patch_size)
    if "vith" in model_name:
        config.hidden_size = 1280
        config.num_hidden_layers = 32
        config.num_attention_heads = 16
        config.layer_norm_eps = 1e-6
        config.mlp_ratio = 4
        config.intermediate_size = 5120
        if model_name == "ijepa_vith16_1k":
            config.image_size = 448
    elif "vitg" in model_name:
        config.hidden_size = 1408
        config.num_hidden_layers = 40
        config.num_attention_heads = 16
        config.layer_norm_eps = 1e-6
        config.mlp_ratio = 48 / 11
        config.intermediate_size = 6144
    else:
        raise ValueError("Model not supported, only supports huge and giant models.")
    return config


@torch.no_grad()
def write_model(model_name, output_dir, safe_serialization, push_to_hub, verify_logits):
    """
    Copy/paste/tweak model's weights to our IJEPA structure.
    """

    # define default IJEPA configuration
    config = get_ijepa_config(model_name)

    checkpoint_mapping = {
        "ijepa_vith14_1k": "https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.14-300e.pth.tar",
        "ijepa_vith14_22k": "https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.h.14-900e.pth.tar",
        "ijepa_vith16_1k": "https://dl.fbaipublicfiles.com/ijepa/IN1K-vit.h.16-448px-300e.pth.tar",
        "ijepa_vitg16_22k": "https://dl.fbaipublicfiles.com/ijepa/IN22K-vit.g.16-600e.pth.tar",
    }

    # Load original checkpoint
    checkpoint_url = checkpoint_mapping[model_name]
    original_state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu")["encoder"]
    original_state_dict = {k.replace("module.", ""): v for k, v in original_state_dict.items()}

    # Rename keys
    state_dict = original_state_dict.copy()
    new_keys = convert_old_keys_to_new_keys(state_dict.keys())
    for old_key, new_key in new_keys.items():
        rename_key(state_dict, old_key, new_key)
    read_in_q_k_v(state_dict, config)

    # load HuggingFace model
    model = IJepaModel(config, add_pooling_layer=False).eval()
    model.load_state_dict(state_dict)
    size = {"height": config.image_size, "width": config.image_size}
    image_processor = ViTImageProcessor(size=size)

    if verify_logits:
        # Check outputs on an image, prepared by ViTImageProcessor
        encoding = image_processor(images=prepare_img(), return_tensors="pt")
        pixel_values = encoding["pixel_values"]
        with torch.no_grad():
            outputs = model(pixel_values)

        expected_slices = {
            "ijepa_vith14_1k": torch.Tensor(
                [[-0.0621, -0.0054, -2.7513], [-0.1952, 0.0909, -3.9536], [0.0942, -0.0331, -1.2833]]
            ),
            "ijepa_vith14_22k": torch.Tensor(
                [[0.0358, -0.0045, -0.2154], [0.0418, -0.0246, 0.0108], [0.2529, -0.0345, -0.0246]]
            ),
            "ijepa_vith16_1k": torch.Tensor(
                [[0.5145, -0.1259, 0.0615], [0.1132, 0.0028, -0.0496], [1.1586, -0.0056, -0.0387]]
            ),
            "ijepa_vitg16_22k": torch.Tensor(
                [[0.0512, -0.0510, -0.0649], [0.1972, 0.0380, -0.0790], [0.1667, -0.0834, -0.1240]]
            ),
        }

        assert torch.allclose(
            expected_slices[model_name],
            outputs.last_hidden_state[0, :3, :3],
            atol=1e-4,
        )

    if output_dir:
        Path(output_dir).mkdir(exist_ok=True)
        print(f"Saving model {model_name} to {output_dir}")
        image_processor.save_pretrained(output_dir, safe_serialization=safe_serialization)
        model.save_pretrained(output_dir, safe_serialization=safe_serialization)

    if push_to_hub:
        image_processor.push_to_hub(repo_id=f"jmtzt/{model_name}", safe_serialization=safe_serialization)
        model.push_to_hub(repo_id=f"jmtzt/{model_name}", safe_serialization=safe_serialization)

    if output_dir:
        del model, state_dict
        gc.collect()
        print("Reloading the model to check if it's saved correctly.")
        IJepaModel.from_pretrained(output_dir, device_map="auto")
        print("Model reloaded successfully.")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="ijepa_vith14_1k",
        type=str,
        choices=[
            "ijepa_vith14_1k",
            "ijepa_vith14_22k",
            "ijepa_vith16_1k",
            "ijepa_vitg16_22k",
        ],
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--safe_serialization", default=True, type=bool, help="Whether or not to save using `safetensors`."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the 🤗 Hub.",
    )
    parser.add_argument(
        "--verify_logits", action="store_false", help="Whether or not to verify logits after conversion."
    )

    parser.set_defaults()
    args = parser.parse_args()
    write_model(args.model_name, args.output_dir, args.safe_serialization, args.push_to_hub, args.verify_logits)


if __name__ == "__main__":
    main()
