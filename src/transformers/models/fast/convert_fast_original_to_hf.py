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

from transformers import AutoConfig, AutoBackbone
from transformers import TextNetConfig
from transformers import FastConfig, FastForSceneTextRecognition
from transformers.models.fast.image_processing_fast import FastImageProcessor



tiny_config_url = "https://raw.githubusercontent.com/czczup/FAST/main/config/fast/nas-configs/fast_tiny.config"
small_config_url = "https://raw.githubusercontent.com/czczup/FAST/main/config/fast/nas-configs/fast_small.config"
base_config_url = "https://raw.githubusercontent.com/czczup/FAST/main/config/fast/nas-configs/fast_base.config"

rename_key_mappings = {
    "module.backbone": "backbone.textnet",
    "first_conv": "stem",
    "bn": "batch_norm",
    "ver": "vertical",
    "hor": "horizontal",
    "module.neck": "neck",
    "module.det_head": "text_detection_head",
}


def prepare_config(size_config_url, size, pooling_size, min_area, bbox_type, loss_bg):
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

    fast_config = FastConfig(
        use_timm_backbone=False,
        backbone_config=textnet_config,
        neck_in_channels=neck_in_channels,
        neck_out_channels=neck_out_channels,
        neck_kernel_size=neck_kernel_size,
        neck_stride=neck_stride,
        neck_dilation=neck_dilation,
        neck_groups=neck_groups,
        head_pooling_size=pooling_size,
        head_dropout_ratio=0.1,
        head_conv_in_channels=config_dict["head"]["conv"]["in_channels"],
        head_conv_out_channels=config_dict["head"]["conv"]["out_channels"],
        head_conv_kernel_size=config_dict["head"]["conv"]["kernel_size"],
        head_conv_stride=config_dict["head"]["conv"]["stride"],
        head_conv_dilation=config_dict["head"]["conv"]["dilation"],
        head_conv_groups=config_dict["head"]["conv"]["groups"],
        head_final_kernel_size=config_dict["head"]["final"]["kernel_size"],
        head_final_stride=config_dict["head"]["final"]["stride"],
        head_final_dilation=config_dict["head"]["final"]["dilation"],
        head_final_groups=config_dict["head"]["final"]["groups"],
        head_final_bias=config_dict["head"]["final"]["bias"],
        head_final_has_shuffle=config_dict["head"]["final"]["has_shuffle"],
        head_final_in_channels=config_dict["head"]["final"]["in_channels"],
        head_final_out_channels=config_dict["head"]["final"]["out_channels"],
        head_final_use_bn=config_dict["head"]["final"]["use_bn"],
        head_final_act_func=config_dict["head"]["final"]["act_func"],
        head_final_dropout_rate=config_dict["head"]["final"]["dropout_rate"],
        head_final_ops_order=config_dict["head"]["final"]["ops_order"],
        min_area=min_area,
        bbox_type=bbox_type,
        loss_bg=loss_bg,
    )

    return fast_config

def get_small_model_config():
    pass


def get_base_model_config():
    pass


def convert_fast_checkpoint(checkpoint_url, checkpoint_config_filename, pytorch_dump_folder_path, save_backbone_separately):
    config_filepath = hf_hub_download(repo_id="Raghavan/fast_model_config_files", filename="fast_model_configs.json")
    # we download the json file for safety reasons
    checkpoint_config_filename_to_json = checkpoint_config_filename.replace(".py", ".json")
    config_model_file_path = hf_hub_download(repo_id="jadechoghari/fast-configs", filename=checkpoint_config_filename_to_json)

    with open(config_filepath) as f:
        content = json.loads(f.read())
    
    with open(config_model_file_path) as f:
        content_model = json.loads(f.read())

    size = content[checkpoint_config_filename]["short_size"]

    #TODO: add logic for py/json files maybe both
    model_config = content_model["model"]
    test_config = content_model.get("test_cfg", None)
    data_config = content_model["data"]
    min_area = 250
    bbox_type = "rect"
    if test_config is not None:
        min_area = test_config.get("min_area", min_area)
        bbox_type = test_config.get("bbox_type", bbox_type)
        loss_bg = test_config.get("loss_emb", None) == "EmbLoss_v2"

    if "tiny" in content[checkpoint_config_filename]["config"]:
        config = prepare_config(
            tiny_config_url, size, model_config["detection_head"]["pooling_size"], min_area, bbox_type, loss_bg
        )

    elif "small" in content[checkpoint_config_filename]["config"]:
        config = prepare_config(
            small_config_url, size, model_config["detection_head"]["pooling_size"], min_area, bbox_type, loss_bg
        )

    else:
        config = prepare_config(
            base_config_url, size, model_config["detection_head"]["pooling_size"], min_area, bbox_type, loss_bg
        )
    
    if "train" in data_config:
        if "short_size" in data_config["train"]:
            size = data_config["train"]["short_size"]

    model = FastForSceneTextRecognition(config)
    fast_image_processor = FastImageProcessor(
        size={"shortest_edge": size},
        min_area=config.min_area,
        bbox_type=config.bbox_type,
        pooling_size=config.head_pooling_size,
    )
    state_dict = torch.hub.load_state_dict_from_url(checkpoint_url, map_location="cpu", check_hash=True)["ema"]
    state_dict_changed = OrderedDict()
    for key in state_dict:
        #TODO: see if we add more
        if "backbone" or "textnet" in key:
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
    pixel_values = fast_image_processor(image, return_tensors="pt").pixel_values
    assert torch.allclose(original_pixel_values, pixel_values[0][0][3][:10], atol=1e-4)

    with torch.no_grad():
        output = model(pixel_values)

    target_sizes = [(image.shape[1], image.shape[2]) for image in pixel_values]
    threshold = 0.85
    text_locations = fast_image_processor.post_process_text_detection(output, target_sizes, threshold, bbox_type="poly")
    #TODO: update assert logic
    # text_locations[0]["bboxes"][0][:10]
    # assert torch.allclose(output["feature_maps"][-1][0][10][12][:10].detach(), expected_slice_backbone, atol=1e-3)
    #TODO: fix the safetensor sharing problem to use safetensors
    # same to remove it, gonna be reassigned in inference
    del model.text_detection_head.final.fused_conv.weight
    model.save_pretrained(pytorch_dump_folder_path)
    if save_backbone_separately:
        model.backbone.save_pretrained(pytorch_dump_folder_path + "/textnet/")
    fast_image_processor.save_pretrained(pytorch_dump_folder_path)
    logging.info("The converted weights are saved here : " + pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/czczup/FAST/releases/download/release/fast_tiny_ic17mlt_640.pth",
        type=str,
        help="URL to the original PyTorch checkpoint (.pth file).",
    )
    parser.add_argument(
        "--checkpoint_config_filename",
        default="fast_tiny_ic17mlt_640.py",
        type=str,
        help="URL to the original PyTorch checkpoint (.pth file).",
    )
    parser.add_argument(
        "--save_backbone_separately",
        default=True,
        type=bool,
        help="whether to assert logits outputs",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default="/home/user/app/transformers/src/transformers/models/fast/output", type=str, help="Path to the folder to output PyTorch model."
    )
    args = parser.parse_args()

    convert_fast_checkpoint(
        args.checkpoint_url,
        args.checkpoint_config_filename,
        args.pytorch_dump_folder_path,
        args.save_backbone_separately,
    )