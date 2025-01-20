# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from pathlib import Path
from typing import List

import regex as re
import torch

from transformers import RelationDetrConfig, RelationDetrForObjectDetection, RelationDetrImageProcessor


# fmt: off
# If a weight needs to be split in two or more keys, use `|` to indicate it. ex:
# r"text_model.layers.(\d+).attention.wqkv.weight": r"language_model.model.layers.\1.self_attn.q|k|v|_proj.weight"
ORIGINAL_TO_CONVERTED_KEY_MAPPING_EXCEPT_BACKBONE = {
    # encoder
    r"transformer.encoder.layers.(\d+).norm1.(weight|bias)":                                                                        r"transformer.encoder.layers.\1.self_attn_layer_norm.\2",
    r"transformer.encoder.layers.(\d+).norm2.(weight|bias)":                                                                        r"transformer.encoder.layers.\1.final_layer_norm.\2",
    r"transformer.(encoder|decoder).layers.(\d+).linear(\d+).(weight|bias)":                                                        r"transformer.\1.layers.\2.fc\3.\4",
    # decoder
    r"transformer.decoder.layers.(\d+).cross_attn.(sampling_offsets|attention_weights|value_proj|output_proj).(weight|bias)":       r"transformer.decoder.layers.\1.encoder_attn.\2.\3",
    r"transformer.decoder.layers.(\d+).norm1.(weight|bias)":                                                                        r"transformer.decoder.layers.\1.encoder_attn_layer_norm.\2",
    r"transformer.decoder.layers.(\d+).norm2.(weight|bias)":                                                                        r"transformer.decoder.layers.\1.self_attn_layer_norm.\2",
    r"transformer.decoder.layers.(\d+).norm3.(weight|bias)":                                                                        r"transformer.decoder.layers.\1.final_layer_norm.\2",
    # embedding layer
    r"transformer.tgt_embed.weight":                                                                                                r"transformer.target_embed.weight",
    r"transformer.hybrid_tgt_embed.weight":                                                                                         r"transformer.hybrid_target_embed.weight",
    r"transformer.level_embeds":                                                                                                    r"transformer.level_embed",
    r"transformer.(.*)":                                                                                                            r"model.\1",
    r"neck.(.*)":                                                                                                                   r"model.neck.\1",
}


ORIGINAL_TO_CONVERTED_KEY_MAPPING_BACKBONE = {
    "resnet": {
        r"backbone.(.*)":                                                   r"model.backbone.model._backbone.\1"
    },
    "swin": {
        r"backbone.0.features.0.0.(weight|bias)":                           r"model.backbone.model._backbone.patch_embed.proj.\1",
        r"backbone.0.features.0.2.(weight|bias)":                           r"model.backbone.model._backbone.patch_embed.norm.\1",
        r"backbone.0.features.(\d+).(\d+).mlp.(\d+).(weight|bias)":         lambda m: f"model.backbone.model._backbone.layers_{int(m.group(1)) // 2}.blocks.{m.group(2)}.mlp.fc{(int(m.group(3)) // 2) + 1}.{m.group(4)}",
        r"backbone.0.features.(\d+).(norm|reduction).(weight|bias)":        lambda m: f"model.backbone.model._backbone.layers_{int(m.group(1)) // 2}.downsample.{m.group(2)}.{m.group(3)}",
        r"backbone.0.features.(\d+).(\d+).attn.relative_position_index":    None,
        r"backbone.0.features.(\d+).(.*)":                                  lambda m: f"model.backbone.model._backbone.layers_{int(m.group(1)) // 2}.blocks.{m.group(2)}",
        r"backbone.0.(.*)":                                                 r"model.backbone.model._backbone.\1",
    },
    "focal": {
        "backbone.0.patch_embed.(proj|norm).(weight|bias)":                r"model.backbone.model._backbone.stem.\1.\2",
        r"backbone.0.layers.(\d+).blocks.(\d+).norm(\d+).(weight|bias)":    r"model.backbone.model._backbone.layers_\1.blocks.\2.norm\3_post.\4",
        r"backbone.0.layers.(\d+).blocks.(\d+).gamma_(\d+)":                r"model.backbone.model._backbone.layers_\1.blocks.\2.ls\3.gamma",
        r"backbone.0.layers.(\d+).blocks.(.*)":                             r"model.backbone.model._backbone.layers_\1.blocks.\2",
        r"backbone.0.layers.(\d+).downsample.(proj|norm).(weight|bias)":    lambda m: f"model.backbone.model._backbone.layers_{int(m.group(1)) + 1}.downsample.{m.group(2)}.{m.group(3)}",
        r"backbone.0.(.*)":                                                 r"model.backbone.model._backbone.\1",
        r"backbone.1.norm(\d+).(weight|bias)":                              r"model.backbone.features_layer_norm.norms.\1.\2",
    }
}
# fmt: on


def convert_old_keys_to_new_keys(state_dict_keys: dict = None, model_name: str = None):
    """
    This function should be applied only once, on the concatenated keys to efficiently rename using
    the key mappings.
    """
    key_mapper = ORIGINAL_TO_CONVERTED_KEY_MAPPING_EXCEPT_BACKBONE
    if model_name is not None:
        if "relation_detr_resnet50" in model_name:
            backbone = "resnet"
        elif "relation_detr_swin_l" in model_name:
            backbone = "swin"
        elif "relation_detr_focal_l" in model_name:
            backbone = "focal"
        else:
            raise ValueError("Unsupported model name!")
        key_mapper.update(ORIGINAL_TO_CONVERTED_KEY_MAPPING_BACKBONE[backbone])

    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in key_mapper.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)  # an empty line
                continue
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict


def split_q_k_v(state_dict):
    # transformer decoder self-attention layers
    for i in range(6):
        # read in weights + bias of input projection layer of self-attention
        in_proj_weight = state_dict.pop(f"model.decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"model.decoder.layers.{i}.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]
    return state_dict


def convert_linear_conv_for_focal(state_dict, model_name):
    if "focal" not in model_name:
        return state_dict

    KEYS_TO_CONVERT = [
        r"model.backbone.model._backbone.layers_(\d+).blocks.(\d+).mlp.fc(\d+).weight",
        r"model.backbone.model._backbone.layers_(\d+).blocks.(\d+).modulation.(proj|f).weight",
    ]
    for key in KEYS_TO_CONVERT:
        for k, v in state_dict.items():
            m = re.match(key, k)
            if m is not None and state_dict[k].dim() == 2:
                # convert linear layer to conv layer
                state_dict[k] = v.view(v.size(0), -1, 1, 1)

    return state_dict


def get_relation_detr_config(model_name: str) -> RelationDetrConfig:
    config = RelationDetrConfig()

    config.num_labels = 91
    config.use_pretrained_backbone = False

    if "relation_detr_resnet50" in model_name:
        config.backbone_kwargs = {"out_indices": [2, 3, 4]}
    elif "relation_detr_swin_l" in model_name:
        config.backbone = "swin_large_patch4_window7_224"
        config.backbone_kwargs = {
            "out_indices": [1, 2, 3],
            "img_size": None,
            "strict_img_size": False,
            "dynamic_img_pad": True,
        }
    elif "relation_detr_focal_l" in model_name:
        config.backbone = "focalnet_large_fl4"
        config.backbone_kwargs = {"out_indices": [0, 1, 2, 3]}
        config.backbone_post_layer_norm = True
        config.num_feature_levels = 5
    else:
        raise ValueError("Unknown model name!")
    return config


def decode_labels(ints: List[int]) -> List[str]:
    """
    Decode a list of int to a list of string, for example: [108, 49, -1, 76, 50, -1, 110, -1] will be decoded as: ["l1", "L2", "n"].
    Each number will be converted to a letter using chr() function in Python, and -1 is used as delimiters to split strings.

    Args:
        ints (List[int]): A list of int to be converted.

    Returns:
        List[str]: A list of string.
    """
    string_list = []
    string = ""
    for number in ints:
        if number != -1:
            string += chr(number)
        else:
            string_list.append(string)
            string = ""
    return string_list


@torch.no_grad()
def convert_relation_detr_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub, repo_id):
    """
    Copy/paste/tweak model's weights to our Relation DETR structure.
    """

    # load default config
    config = get_relation_detr_config(model_name)

    # load original model from torch hub
    model_name_to_checkpoint_url = {
        "relation_detr_resnet50_coco_1x": "https://github.com/xiuqhou/Relation-DETR/releases/download/v1.0.0/relation_detr_resnet50_800_1333_coco_1x.pth",
        "relation_detr_resnet50_coco_2x": "https://github.com/xiuqhou/Relation-DETR/releases/download/v1.0.0/relation_detr_resnet50_800_1333_coco_2x.pth",
        "relation_detr_swin_l_coco_1x": "https://github.com/xiuqhou/Relation-DETR/releases/download/v1.0.0/relation_detr_swin_l_800_1333_coco_1x.pth",
        "relation_detr_swin_l_coco_2x": "https://github.com/xiuqhou/Relation-DETR/releases/download/v1.0.0/relation_detr_swin_l_800_1333_coco_2x.pth",
        "relation_detr_focal_l_o365_4e": "https://github.com/xiuqhou/Relation-DETR/releases/download/v1.0.0/relation_detr_focalnet_large_lrf_fl4_800_1333_o365_4e.pth",
        "relation_detr_focal_l_o365_4e_coco_2x": "https://github.com/xiuqhou/Relation-DETR/releases/download/v1.0.0/relation_detr_focalnet_large_lrf_fl4_1200_2000_o365_4e-coco_2x.pth",
    }

    print(f"Converting model {model_name}...")
    state_dict = torch.hub.load_state_dict_from_url(model_name_to_checkpoint_url[model_name], map_location="cpu")

    # obtain label information from original state_dict
    labels = state_dict.pop("_classes_")
    labels = decode_labels(labels)
    config.id2label = dict(enumerate(labels))
    config.label2id = {label: i for i, label in config.id2label.items()}

    # rename keys
    all_keys = list(state_dict.keys())
    new_keys = convert_old_keys_to_new_keys(all_keys, model_name=model_name)
    for old_key, new_key in new_keys.items():
        if new_key == "":
            state_dict.pop(old_key)
        elif old_key != new_key:
            state_dict[new_key] = state_dict.pop(old_key)

    # process for focalnet
    convert_linear_conv_for_focal(state_dict, model_name)

    # query, key and value matrices need special treatment
    split_q_k_v(state_dict)

    # finally, create HuggingFace model and load state dict
    model = RelationDetrForObjectDetection(config)
    model.load_state_dict(state_dict, strict=True)
    print("Checkpoint loaded successfully.")

    # load image processor
    image_processor = RelationDetrImageProcessor(size_divisor=32)

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        print(f"Saving image processor to {pytorch_dump_folder_path}")
        image_processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        # Upload model, image processor and config to the hub
        print("Uploading PyTorch model and image processor to the hub...")
        config.push_to_hub(
            repo_id=repo_id,
            commit_message="Add config from convert_relation_detr_original_pytorch_checkpoint_to_pytorch.py",
        )
        model.push_to_hub(
            repo_id=repo_id,
            commit_message="Add model from convert_relation_detr_original_pytorch_checkpoint_to_pytorch.py",
        )
        image_processor.push_to_hub(
            repo_id=repo_id,
            commit_message="Add image processor from convert_relation_detr_original_pytorch_checkpoint_to_pytorch.py",
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="relation-detr-resnet50",
        type=str,
        help="model_name of the checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to the hub or not.")
    parser.add_argument(
        "--repo_id",
        type=str,
        help="repo_id where the model will be pushed to.",
    )
    args = parser.parse_args()
    convert_relation_detr_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub, args.repo_id)


if __name__ == "__main__":
    main()
