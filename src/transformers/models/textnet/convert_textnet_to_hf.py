# coding=utf-8
# Copyright 2024 the Fast authors and The HuggingFace Inc. team. All rights reserved.
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

import argparse
import json
import logging
import re
from collections import OrderedDict

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import TextNetBackbone, TextNetConfig, TextNetImageProcessor


tiny_config_url = "https://raw.githubusercontent.com/czczup/FAST/main/config/fast/nas-configs/fast_tiny.config"
small_config_url = "https://raw.githubusercontent.com/czczup/FAST/main/config/fast/nas-configs/fast_small.config"
base_config_url = "https://raw.githubusercontent.com/czczup/FAST/main/config/fast/nas-configs/fast_base.config"

rename_key_mappings = {
    "module.backbone": "textnet",
    "first_conv": "stem",
    "bn": "batch_norm",
    "ver": "vertical",
    "hor": "horizontal",
}


def prepare_config(size_config_url, size):
    config_dict = json.loads(requests.get(size_config_url).text)

    backbone_config = {}
    for stage_ix in range(1, 5):
        stage_config = config_dict[f"stage{stage_ix}"]

        merged_dict = {}

        # Iterate through the list of dictionaries
        for layer in stage_config:
            for key, value in layer.items():
                if key != "name":
                    # Check if the key is already in the merged_dict
                    if key in merged_dict:
                        merged_dict[key].append(value)
                    else:
                        # If the key is not in merged_dict, create a new list with the value
                        merged_dict[key] = [value]
        backbone_config[f"stage{stage_ix}"] = merged_dict

    neck_in_channels = []
    neck_out_channels = []
    neck_kernel_size = []
    neck_stride = []
    neck_dilation = []
    neck_groups = []

    for i in range(1, 5):
        layer_key = f"reduce_layer{i}"
        layer_dict = config_dict["neck"].get(layer_key)

        if layer_dict:
            # Append values to the corresponding lists
            neck_in_channels.append(layer_dict["in_channels"])
            neck_out_channels.append(layer_dict["out_channels"])
            neck_kernel_size.append(layer_dict["kernel_size"])
            neck_stride.append(layer_dict["stride"])
            neck_dilation.append(layer_dict["dilation"])
            neck_groups.append(layer_dict["groups"])

    textnet_config = TextNetConfig(
        stem_kernel_size=config_dict["first_conv"]["kernel_size"],
        stem_stride=config_dict["first_conv"]["stride"],
        stem_num_channels=config_dict["first_conv"]["in_channels"],
        stem_out_channels=config_dict["first_conv"]["out_channels"],
        stem_act_func=config_dict["first_conv"]["act_func"],
        conv_layer_kernel_sizes=[
            backbone_config["stage1"]["kernel_size"],
            backbone_config["stage2"]["kernel_size"],
            backbone_config["stage3"]["kernel_size"],
            backbone_config["stage4"]["kernel_size"],
        ],
        conv_layer_strides=[
            backbone_config["stage1"]["stride"],
            backbone_config["stage2"]["stride"],
            backbone_config["stage3"]["stride"],
            backbone_config["stage4"]["stride"],
        ],
        hidden_sizes=[
            config_dict["first_conv"]["out_channels"],
            backbone_config["stage1"]["out_channels"][-1],
            backbone_config["stage2"]["out_channels"][-1],
            backbone_config["stage3"]["out_channels"][-1],
            backbone_config["stage4"]["out_channels"][-1],
        ],
        out_features=["stage1", "stage2", "stage3", "stage4"],
        out_indices=[1, 2, 3, 4],
    )

    return textnet_config


def convert_textnet_checkpoint(checkpoint_url, checkpoint_config_filename, pytorch_dump_folder_path):
    config_filepath = hf_hub_download(repo_id="Raghavan/fast_model_config_files", filename="fast_model_configs.json")

    with open(config_filepath) as f:
        content = json.loads(f.read())

    size = content[checkpoint_config_filename]["short_size"]

    if "tiny" in content[checkpoint_config_filename]["config"]:
        config = prepare_config(tiny_config_url, size)
        expected_slice_backbone = torch.tensor(
            [0.0000, 0.0000, 0.0000, 0.0000, 0.5300, 0.0000, 0.0000, 0.0000, 0.0000, 1.1221]
        )
    elif "small" in content[checkpoint_config_filename]["config"]:
        config = prepare_config(small_config_url, size)
        expected_slice_backbone = torch.tensor(
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1394]
        )
    else:
        config = prepare_config(base_config_url, size)
        expected_slice_backbone = torch.tensor(
            [0.9210, 0.6099, 0.0000, 0.0000, 0.0000, 0.0000, 3.2207, 2.6602, 1.8925, 0.0000]
        )

    model = TextNetBackbone(config)
    textnet_image_processor = TextNetImageProcessor(size={"shortest_edge": size})
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu", check_hash=True)["ema"]
    state_dict_changed = OrderedDict()
    for key in state_dict:
        if "backbone" in key:
            val = state_dict[key]
            new_key = key
            for search, replacement in rename_key_mappings.items():
                if search in new_key:
                    new_key = new_key.replace(search, replacement)

            pattern = r"textnet\.stage(\d)"

            def adjust_stage(match):
                stage_number = int(match.group(1)) - 1
                return f"textnet.encoder.stages.{stage_number}.stage"

            # Using regex to find and replace the pattern in the string
            new_key = re.sub(pattern, adjust_stage, new_key)
            state_dict_changed[new_key] = val
    model.load_state_dict(state_dict_changed)
    model.eval()

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

    original_pixel_values = torch.tensor(
        [0.1939, 0.3481, 0.4166, 0.3309, 0.4508, 0.4679, 0.4851, 0.4851, 0.3309, 0.4337]
    )
    pixel_values = textnet_image_processor(image, return_tensors="pt").pixel_values

    assert torch.allclose(original_pixel_values, pixel_values[0][0][3][:10], atol=1e-4)

    with torch.no_grad():
        output = model(pixel_values)

    assert torch.allclose(output["feature_maps"][-1][0][10][12][:10].detach(), expected_slice_backbone, atol=1e-3)

    model.save_pretrained(pytorch_dump_folder_path)
    textnet_image_processor.save_pretrained(pytorch_dump_folder_path)
    logging.info("The converted weights are saved here : " + pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/czczup/FAST/releases/download/release/fast_base_ic17mlt_640.pth",
        type=str,
        help="URL to the original PyTorch checkpoint (.pth file).",
    )
    parser.add_argument(
        "--checkpoint_config_filename",
        default="fast_base_ic17mlt_640.py",
        type=str,
        help="URL to the original PyTorch checkpoint (.pth file).",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the folder to output PyTorch model."
    )
    args = parser.parse_args()

    convert_textnet_checkpoint(
        args.checkpoint_url,
        args.checkpoint_config_filename,
        args.pytorch_dump_folder_path,
    )
