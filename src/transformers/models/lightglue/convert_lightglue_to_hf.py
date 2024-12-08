# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import gc
import os
import re
from typing import List

import torch

from transformers import (
    AutoModelForKeypointDetection,
    LightGlueForKeypointMatching,
    LightGlueImageProcessor,
)
from transformers.models.lightglue.configuration_lightglue import LightGlueConfig


def verify_model_outputs(model):
    from tests.models.lightglue.test_modeling_lightglue import prepare_imgs

    device = "cuda"
    images = prepare_imgs()
    preprocessor = LightGlueImageProcessor()
    inputs = preprocessor(images=images, return_tensors="pt").to(device)
    model.to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    predicted_matches_values = outputs.matches[0, 0, 20:30]
    predicted_matching_scores_values = outputs.matching_scores[0, 0, 20:30]

    predicted_number_of_matches = torch.sum(outputs.matches[0][0] != -1).item()

    expected_max_number_keypoints = 918
    expected_matches_shape = torch.Size((len(images), 2, expected_max_number_keypoints))
    expected_matching_scores_shape = torch.Size((len(images), 2, expected_max_number_keypoints))

    expected_matches_values = torch.tensor([-1, 25, -1, -1, -1, -1, -1, -1, -1, -1], dtype=torch.int64).to(device)
    expected_matching_scores_values = torch.tensor([0, 0.1502, 0, 0, 0.0031, 0, 0, 0, 0, 0]).to(device)

    expected_number_of_matches = 147

    assert outputs.matches.shape == expected_matches_shape
    assert outputs.matching_scores.shape == expected_matching_scores_shape

    assert torch.allclose(predicted_matches_values, expected_matches_values, atol=1e-4)
    assert torch.allclose(predicted_matching_scores_values, expected_matching_scores_values, atol=1e-4)

    assert predicted_number_of_matches == expected_number_of_matches


ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"posenc.Wr": r"positional_encoder.projector",
    r"self_attn.(\d+).Wqkv": r"transformer_layers.\1.self_attention_block.Wqkv",
    r"self_attn.(\d+).out_proj": r"transformer_layers.\1.self_attention_block.output.dense",
    r"self_attn.(\d+).ffn.0": r"transformer_layers.\1.self_attention_block.output.mlp.dense",
    r"self_attn.(\d+).ffn.1": r"transformer_layers.\1.self_attention_block.output.mlp.layer_norm",
    r"self_attn.(\d+).ffn.3": r"transformer_layers.\1.self_attention_block.output.mlp.output",
    r"cross_attn.(\d+).to_qk": r"transformer_layers.\1.cross_attention_block.to_qk",
    r"cross_attn.(\d+).to_v": r"transformer_layers.\1.cross_attention_block.self.value",
    r"cross_attn.(\d+).to_out": r"transformer_layers.\1.cross_attention_block.output.dense",
    r"cross_attn.(\d+).ffn.0": r"transformer_layers.\1.cross_attention_block.output.mlp.dense",
    r"cross_attn.(\d+).ffn.1": r"transformer_layers.\1.cross_attention_block.output.mlp.layer_norm",
    r"cross_attn.(\d+).ffn.3": r"transformer_layers.\1.cross_attention_block.output.mlp.output",
    r"log_assignment.(\d+).matchability": r"match_assignment_layers.\1.matchability",
    r"log_assignment.(\d+).final_proj": r"match_assignment_layers.\1.final_projection",
    r"token_confidence.(\d+).token.0": r"token_confidence.\1.token",
}


def convert_old_keys_to_new_keys(state_dict_keys: List[str]):
    """
    This function should be applied only once, on the concatenated keys to efficiently rename using
    the key mappings.
    """
    output_dict = {}
    if state_dict_keys is not None:
        old_text = "\n".join(state_dict_keys)
        new_text = old_text
        for pattern, replacement in ORIGINAL_TO_CONVERTED_KEY_MAPPING.items():
            if replacement is None:
                new_text = re.sub(pattern, "", new_text)  # an empty line
                continue
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict


def add_keypoint_detector_state_dict(lightglue_state_dict):
    keypoint_detector = AutoModelForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
    keypoint_detector_state_dict = keypoint_detector.state_dict()
    for k, v in keypoint_detector_state_dict.items():
        lightglue_state_dict[f"keypoint_detector.{k}"] = v
    return lightglue_state_dict


def split_weights(state_dict):
    for i in range(9):
        Wqkv_weight = state_dict.pop(f"transformer_layers.{i}.self_attention_block.Wqkv.weight")
        Wqkv_bias = state_dict.pop(f"transformer_layers.{i}.self_attention_block.Wqkv.bias")
        Wqkv_weight = Wqkv_weight.reshape(256, 3, 256)
        Wqkv_bias = Wqkv_bias.reshape(256, 3)
        query_weight, key_weight, value_weight = Wqkv_weight[:, 0], Wqkv_weight[:, 1], Wqkv_weight[:, 2]
        query_bias, key_bias, value_bias = Wqkv_bias[:, 0], Wqkv_bias[:, 1], Wqkv_bias[:, 2]
        state_dict[f"transformer_layers.{i}.self_attention_block.self.query.weight"] = query_weight
        state_dict[f"transformer_layers.{i}.self_attention_block.self.key.weight"] = key_weight
        state_dict[f"transformer_layers.{i}.self_attention_block.self.value.weight"] = value_weight
        state_dict[f"transformer_layers.{i}.self_attention_block.self.query.bias"] = query_bias
        state_dict[f"transformer_layers.{i}.self_attention_block.self.key.bias"] = key_bias
        state_dict[f"transformer_layers.{i}.self_attention_block.self.value.bias"] = value_bias

        to_qk_weight = state_dict.pop(f"transformer_layers.{i}.cross_attention_block.to_qk.weight")
        to_qk_bias = state_dict.pop(f"transformer_layers.{i}.cross_attention_block.to_qk.bias")
        state_dict[f"transformer_layers.{i}.cross_attention_block.self.query.weight"] = to_qk_weight
        state_dict[f"transformer_layers.{i}.cross_attention_block.self.query.bias"] = to_qk_bias
        state_dict[f"transformer_layers.{i}.cross_attention_block.self.key.weight"] = to_qk_weight
        state_dict[f"transformer_layers.{i}.cross_attention_block.self.key.bias"] = to_qk_bias

    return state_dict


@torch.no_grad()
def write_model(
    model_path,
    checkpoint_url,
    organization,
    safe_serialization=True,
    push_to_hub=False,
):
    os.makedirs(model_path, exist_ok=True)

    # ------------------------------------------------------------
    # LightGlue config
    # ------------------------------------------------------------

    config = LightGlueConfig(
        descriptor_dim=256,
        num_hidden_layers=9,
        num_attention_heads=4,
    )
    config.architectures = ["LightGlueForKeypointMatching"]
    config.save_pretrained(model_path)
    print("Model config saved successfully...")

    # ------------------------------------------------------------
    # Convert weights
    # ------------------------------------------------------------

    print(f"Fetching all parameters from the checkpoint at {checkpoint_url}...")
    original_state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)

    print("Converting model...")
    all_keys = list(original_state_dict.keys())
    new_keys = convert_old_keys_to_new_keys(all_keys)

    state_dict = {}
    for key in all_keys:
        new_key = new_keys[key]
        state_dict[new_key] = original_state_dict.pop(key).contiguous().clone()

    del original_state_dict
    gc.collect()
    state_dict = split_weights(state_dict)
    state_dict = add_keypoint_detector_state_dict(state_dict)

    print("Loading the checkpoint in a SuperGlue model...")
    with torch.device("cuda"):
        model = LightGlueForKeypointMatching(config)
    model.load_state_dict(state_dict, strict=False)
    print("Checkpoint loaded successfully...")
    del model.config._name_or_path

    print("Saving the model...")
    model.save_pretrained(model_path, safe_serialization=safe_serialization)
    del state_dict, model

    # Safety check: reload the converted model
    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    model = LightGlueForKeypointMatching.from_pretrained(model_path)
    print("Model reloaded successfully.")

    model_name = "lightglue"
    if "superpoint" in checkpoint_url:
        model_name += "_superpoint"
    print("Checking the model outputs...")
    verify_model_outputs(model)
    print("Model outputs verified successfully.")

    if push_to_hub:
        print("Pushing model to the hub...")
        model.push_to_hub(
            repo_id=f"{organization}/{model_name}",
            commit_message="Add model",
        )
        config.push_to_hub(repo_id=f"{organization}/{model_name}", commit_message="Add config")

    write_image_processor(model_path, model_name, organization, push_to_hub=push_to_hub)


def write_image_processor(save_dir, model_name, organization, push_to_hub=False):
    if "superpoint" in model_name:
        image_processor = LightGlueImageProcessor(do_grayscale=True)
    else:
        image_processor = LightGlueImageProcessor()
    image_processor.save_pretrained(save_dir)

    if push_to_hub:
        print("Pushing image processor to the hub...")
        image_processor.push_to_hub(
            repo_id=f"{organization}/{model_name}",
            commit_message="Add image processor",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--checkpoint_url",
        default="https://github.com/cvg/LightGlue/releases/download/v0.1_arxiv/superpoint_lightglue.pth",
        type=str,
        help="URL of the original LightGlue checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument("--save_model", action="store_true", help="Save model to local")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push model and image preprocessor to the hub",
    )
    parser.add_argument(
        "--organization",
        default="stevenbucaille",
        type=str,
        help="Hub organization in which you want the model to be uploaded.",
    )

    args = parser.parse_args()
    write_model(
        args.pytorch_dump_folder_path,
        args.checkpoint_url,
        args.organization,
        safe_serialization=True,
        push_to_hub=args.push_to_hub,
    )
