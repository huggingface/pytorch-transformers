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
from pathlib import Path
from typing import Tuple

import torch
from datasets import load_dataset

from transformers import (
    AutoConfig,
    AutoModelForKeypointDetection,
    LightGlueForKeypointMatching,
    LightGlueImageProcessor,
)

# SuperGlueConfig,
# SuperGlueForKeypointMatching,
# SuperGlueImageProcessor,
from transformers.models.lightglue.configuration_lightglue import LightGlueConfig
from transformers.models.superpoint.modeling_superpoint import SuperPointKeypointDescriptionOutput
from transformers.utils import is_flash_attn_2_available


def get_lightglue_config():
    config = LightGlueConfig()

    return config


def create_rename_keys(config, state_dict):
    rename_keys = []

    # positional encoder
    rename_keys.append(
        (
            "posenc.Wr.weight",
            "positional_encoder.projector.weight",
        )
    )

    for i in range(config.num_layers):
        # self attention blocks
        rename_keys.append(
            (
                f"self_attn.{i}.Wqkv.weight",
                f"transformer_layers.{i}.self_attention_block.Wqkv.weight",
            )
        )
        rename_keys.append(
            (
                f"self_attn.{i}.Wqkv.bias",
                f"transformer_layers.{i}.self_attention_block.Wqkv.bias",
            )
        )
        rename_keys.append(
            (
                f"self_attn.{i}.out_proj.weight",
                f"transformer_layers.{i}.self_attention_block.output_projection.weight",
            )
        )
        rename_keys.append(
            (
                f"self_attn.{i}.out_proj.bias",
                f"transformer_layers.{i}.self_attention_block.output_projection.bias",
            )
        )
        for j in [0, 1, 3]:
            rename_keys.append(
                (
                    f"self_attn.{i}.ffn.{j}.weight",
                    f"transformer_layers.{i}.self_attention_block.ffn.{j}.weight",
                )
            )
            rename_keys.append(
                (
                    f"self_attn.{i}.ffn.{j}.bias",
                    f"transformer_layers.{i}.self_attention_block.ffn.{j}.bias",
                )
            )

        # cross attention blocks
        rename_keys.append(
            (
                f"cross_attn.{i}.to_qk.weight",
                f"transformer_layers.{i}.cross_attention_block.to_qk.weight",
            )
        )
        rename_keys.append(
            (
                f"cross_attn.{i}.to_qk.bias",
                f"transformer_layers.{i}.cross_attention_block.to_qk.bias",
            )
        )
        rename_keys.append(
            (
                f"cross_attn.{i}.to_v.weight",
                f"transformer_layers.{i}.cross_attention_block.to_v.weight",
            )
        )
        rename_keys.append(
            (
                f"cross_attn.{i}.to_v.bias",
                f"transformer_layers.{i}.cross_attention_block.to_v.bias",
            )
        )
        rename_keys.append(
            (
                f"cross_attn.{i}.to_out.weight",
                f"transformer_layers.{i}.cross_attention_block.to_out.weight",
            )
        )
        rename_keys.append(
            (
                f"cross_attn.{i}.to_out.bias",
                f"transformer_layers.{i}.cross_attention_block.to_out.bias",
            )
        )

        for j in [0, 1, 3]:
            rename_keys.append(
                (
                    f"cross_attn.{i}.ffn.{j}.weight",
                    f"transformer_layers.{i}.cross_attention_block.ffn.{j}.weight",
                )
            )
            rename_keys.append(
                (
                    f"cross_attn.{i}.ffn.{j}.bias",
                    f"transformer_layers.{i}.cross_attention_block.ffn.{j}.bias",
                )
            )

        # Match assignment layers
        rename_keys.append(
            (
                f"log_assignment.{i}.matchability.weight",
                f"match_assignment_layers.{i}.matchability.weight",
            )
        )
        rename_keys.append(
            (
                f"log_assignment.{i}.matchability.bias",
                f"match_assignment_layers.{i}.matchability.bias",
            )
        )
        rename_keys.append(
            (
                f"log_assignment.{i}.final_proj.weight",
                f"match_assignment_layers.{i}.final_projection.weight",
            )
        )
        rename_keys.append(
            (
                f"log_assignment.{i}.final_proj.bias",
                f"match_assignment_layers.{i}.final_projection.bias",
            )
        )
        if i < config.num_layers - 1:
            rename_keys.append(
                (
                    f"token_confidence.{i}.token.0.weight",
                    f"token_confidence.{i}.token.weight",
                )
            )
            rename_keys.append(
                (
                    f"token_confidence.{i}.token.0.bias",
                    f"token_confidence.{i}.token.bias",
                )
            )

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def add_keypoint_detector_state_dict(lightglue_state_dict, keypoint_detector_state_dict):
    for k, v in keypoint_detector_state_dict.items():
        lightglue_state_dict[f"keypoint_detector.{k}"] = v
    return lightglue_state_dict


def prepare_imgs_for_image_processor():
    dataset = load_dataset("stevenbucaille/image_matching_fixtures", split="train")
    return [[dataset[0]["image"], dataset[1]["image"]]]


def extract_keypoint_information_from_image_point_description_output(
    output: SuperPointKeypointDescriptionOutput, i: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = torch.nonzero(output.mask[i]).squeeze()
    keypoints = torch.unsqueeze(output.keypoints[i][indices], dim=0)
    descriptors = torch.unsqueeze(output.descriptors[i][indices], dim=0)
    scores = torch.unsqueeze(output.scores[i][indices], dim=0)
    return keypoints, descriptors, scores


@torch.no_grad()
def convert_lightglue_checkpoint(checkpoint_url, pytorch_dump_folder_path, save_model, push_to_hub):
    """
    Copy/paste/tweak model's weights to our SuperPoint structure.
    Also test the model with the image processor and other methods of reading the images.
    """
    torch.set_printoptions(precision=10)
    keypoint_detector_config = AutoConfig.from_pretrained("magic-leap-community/superpoint")
    keypoint_detector_config.max_keypoints = 2048
    keypoint_detector_config.keypoint_threshold = 0.0005
    lightglue_config = get_lightglue_config()
    lightglue_config.keypoint_detector_config = keypoint_detector_config

    keypoint_detector = AutoModelForKeypointDetection.from_pretrained("magic-leap-community/superpoint")
    keypoint_detector.config = keypoint_detector_config
    keypoint_detector.to("cuda")
    keypoint_detector.eval()
    keypoint_detector_state_dict = keypoint_detector.state_dict()

    print("Downloading original model from checkpoint...")
    original_lightglue_state_dict = torch.hub.load_state_dict_from_url(checkpoint_url)

    print("Converting model parameters...")
    rename_keys = create_rename_keys(lightglue_config, original_lightglue_state_dict)
    new_lightglue_state_dict = original_lightglue_state_dict.copy()
    for src, dest in rename_keys:
        rename_key(new_lightglue_state_dict, src, dest)
    new_lightglue_state_dict = add_keypoint_detector_state_dict(new_lightglue_state_dict, keypoint_detector_state_dict)

    expected_number_of_matches = 264

    # eager mode
    lightglue_config._attn_implementation = "eager"
    model = LightGlueForKeypointMatching(lightglue_config)
    model.load_state_dict(new_lightglue_state_dict, strict=False, assign=False)
    model.to("cuda")
    model.eval()
    print("Successfully loaded weights in the model")

    ## USE REGULAR IMAGE PROCESSOR FOR INFERENCE
    images = prepare_imgs_for_image_processor()
    preprocessor = LightGlueImageProcessor()
    inputs = preprocessor(images=images, return_tensors="pt")
    inputs.to("cuda")
    torch.save(inputs["pixel_values"], "lightglue_input.pth")

    output = model(**inputs, return_dict=True)

    print("Number of matching keypoints using image processor")
    print(torch.sum(output.matches[0][0] != -1))
    print(output.matches[0][0][:10])
    assert expected_number_of_matches == torch.sum(output.matches[0][0] != -1)

    if save_model:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        preprocessor.save_pretrained(pytorch_dump_folder_path)

        if push_to_hub:
            print("Pushing model to the hub...")
            model_name = "lightglue"

            model.push_to_hub(
                repo_id=f"stevenbucaille/{model_name}",
                commit_message="Add model",
            )
            preprocessor.push_to_hub(model_name)

    # sdpa mode
    lightglue_config._attn_implementation = "sdpa"
    model = LightGlueForKeypointMatching(lightglue_config)
    model.load_state_dict(new_lightglue_state_dict, strict=False, assign=False)
    model.to("cuda")
    model.eval()
    print("Successfully loaded weights in the model")

    ## USE REGULAR IMAGE PROCESSOR FOR INFERENCE
    images = prepare_imgs_for_image_processor()
    preprocessor = LightGlueImageProcessor()
    inputs = preprocessor(images=images, return_tensors="pt")
    inputs.to("cuda")
    torch.save(inputs["pixel_values"], "lightglue_input.pth")

    output = model(**inputs, return_dict=True)

    print("Number of matching keypoints using image processor")
    print(torch.sum(output.matches[0][0] != -1))
    print(output.matches[0][0][:10])

    assert expected_number_of_matches == torch.sum(output.matches[0][0] != -1)

    if is_flash_attn_2_available():
        # flash attention mode
        lightglue_config._attn_implementation = "flash_attention_2"
        with torch.autocast(device_type="cuda"):
            model = LightGlueForKeypointMatching(lightglue_config)
            model.load_state_dict(new_lightglue_state_dict, strict=False, assign=False)
            model.to("cuda")
            model.eval()
            print("Successfully loaded weights in the model")

            ## USE REGULAR IMAGE PROCESSOR FOR INFERENCE
            images = prepare_imgs_for_image_processor()
            preprocessor = LightGlueImageProcessor()
            inputs = preprocessor(images=images, return_tensors="pt")
            inputs.to("cuda")
            torch.save(inputs["pixel_values"], "lightglue_input.pth")
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                output = model(**inputs, return_dict=True)

            print("Number of matching keypoints using image processor")
            print(torch.sum(output.matches[0][0] != -1))
            print(output.matches[0][0][:10])

            assert expected_number_of_matches == torch.sum(output.matches[0][0] != -1)


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

    args = parser.parse_args()
    convert_lightglue_checkpoint(
        args.checkpoint_url,
        args.pytorch_dump_folder_path,
        args.save_model,
        args.push_to_hub,
    )
