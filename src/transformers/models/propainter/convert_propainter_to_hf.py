# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the S-Lab License, Version 1.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/sczhou/ProPainter/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted weights from original model code at https://github.com/sczhou/ProPainter

import argparse
import os
import re

import numpy as np
import torch
from datasets import load_dataset

from transformers import (
    ProPainterConfig,
    ProPainterModel,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


rename_rules_optical_flow = [
    (r"fnet", r"feature_network"),
    (r"cnet", r"context_network"),
    (r"update_block", r"update_block"),
    (r"module\.(fnet|cnet|update_block)", r"optical_flow_model.\1"),
    (r"layer(\d+)\.(\d+)", lambda m: f"resblocks.{(int(m.group(1)) - 1) * 2 + int(m.group(2))}"),
    (r"convc", "conv_corr"),
    (r"convf", "conv_flow"),
]

rename_rules_flow_completion = [
    (r"downsample", r"flow_completion_net.downsample"),
    (r"encoder1", r"flow_completion_net.encoder1"),
    (r"encoder2", r"flow_completion_net.encoder2"),
    (r"decoder1", r"flow_completion_net.decoder1"),
    (r"decoder2", r"flow_completion_net.decoder2"),
    (r"upsample", r"flow_completion_net.upsample"),
    (r"mid_dilation", r"flow_completion_net.intermediate_dilation"),
    (
        r"feat_prop_module\.deform_align\.backward_",
        r"flow_completion_net.feature_propagation_module.deform_align.backward_",
    ),
    (
        r"feat_prop_module\.deform_align\.forward_",
        r"flow_completion_net.feature_propagation_module.deform_align.forward_",
    ),
    (r"feat_prop_module\.backbone\.backward_", r"flow_completion_net.feature_propagation_module.backbone.backward_"),
    (r"feat_prop_module\.backbone\.forward_", r"flow_completion_net.feature_propagation_module.backbone.forward_"),
    (r"feat_prop_module\.fusion", r"flow_completion_net.feature_propagation_module.fusion"),
    (r"edgeDetector\.projection", r"flow_completion_net.edgeDetector.projection"),
    (r"edgeDetector\.mid_layer", r"flow_completion_net.edgeDetector.intermediate_layer"),
    (r"edgeDetector\.out_layer", r"flow_completion_net.edgeDetector.out_layer"),
]

rename_rules_inpaint_generator = [
    (r"encoder", r"inpaint_generator.encoder"),
    (r"decoder", r"inpaint_generator.decoder"),
    (r"ss", r"inpaint_generator.soft_split"),
    (r"sc", r"inpaint_generator.soft_comp"),
    (r"feat_prop_module\.", r"inpaint_generator.feature_propagation_module."),
    (r"transformers\.transformer\.", r"inpaint_generator.transformers.transformer."),
    (r"norm", r"layer_norm"),
]


def apply_rename_rules(old_key, rules):
    """Apply rename rules using regex substitutions."""
    new_key = old_key
    for pattern, replacement in rules:
        new_key = re.sub(pattern, replacement, new_key)
    return new_key


def map_keys(old_keys, module):
    key_mapping = {}

    # Apply the appropriate rename rules based on the module type
    if module == "optical_flow":
        rename_rules = rename_rules_optical_flow
    elif module == "flow_completion":
        rename_rules = rename_rules_flow_completion
    else:
        rename_rules = rename_rules_inpaint_generator

    for old_key in old_keys:
        new_key = apply_rename_rules(old_key, rename_rules)
        key_mapping[new_key] = old_key

    return key_mapping


def rename_key(state_dict, old_key, new_key):
    if old_key in state_dict:
        state_dict[new_key] = state_dict.pop(old_key)


def create_new_state_dict(combined_state_dict, original_state_dict, key_mapping):
    for new_key, old_key in key_mapping.items():
        rename_key(original_state_dict, old_key, new_key)
        combined_state_dict[new_key] = original_state_dict[new_key]


def prepare_input():
    ds = load_dataset("ruffy369/propainter-object-removal")
    ds_images = ds["train"]["image"]
    num_frames = len(ds_images) // 2
    video = [np.array(ds_images[i]) for i in range(num_frames)]
    # stack to convert H,W mask frame to compatible H,W,C frame
    masks = [np.stack([np.array(ds_images[i])], axis=-1) for i in range(num_frames, 2 * num_frames)]
    return video, masks


@torch.no_grad()
def convert_propainter_checkpoint(args):
    combined_state_dict = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Download the original checkpoint
    original_state_dict_optical_flow = torch.hub.load_state_dict_from_url(
        args.optical_flow_checkpoint_url, map_location="cpu"
    )
    original_state_dict_flow_completion = torch.hub.load_state_dict_from_url(
        args.flow_completion_checkpoint_url, map_location="cpu"
    )
    original_state_dict_inpaint_generator = torch.hub.load_state_dict_from_url(
        args.inpaint_generator_checkpoint_url, map_location="cpu"
    )

    key_mapping_optical_flow = map_keys(list(original_state_dict_optical_flow.keys()), "optical_flow")

    key_mapping_flow_completion = map_keys(list(original_state_dict_flow_completion.keys()), "flow_completion")

    key_mapping_inpaint_generator = map_keys(list(original_state_dict_inpaint_generator.keys()), "inpaint_generator")

    # Create new state dict with updated keys for optical flow model
    create_new_state_dict(combined_state_dict, original_state_dict_optical_flow, key_mapping_optical_flow)

    # Create new state dict with updated keys for flow completion network
    create_new_state_dict(
        combined_state_dict,
        original_state_dict_flow_completion,
        key_mapping_flow_completion,
    )

    # Create new state dict with updated keys for propainter inpaint generator
    create_new_state_dict(
        combined_state_dict,
        original_state_dict_inpaint_generator,
        key_mapping_inpaint_generator,
    )

    dummy_checkpoint_path = os.path.join(args.pytorch_dump_folder_path, "pytorch_model.bin")
    torch.save(combined_state_dict, dummy_checkpoint_path)

    # Load created new state dict after weights conversion (model.load_state_dict wasn't used because some parameters in the model are initialised
    # instead of being loaded from any pretrained model and error occurs with `load_state_dict`)
    model = (
        ProPainterModel(ProPainterConfig())
        .from_pretrained(f"{args.pytorch_dump_folder_path}/", local_files_only=True)
        .to(device)
    )
    model.eval()

    if os.path.exists(dummy_checkpoint_path):
        os.remove(dummy_checkpoint_path)

    if args.pytorch_dump_folder_path is not None:
        print(f"Saving model for {args.model_name} to {args.pytorch_dump_folder_path}")
        model.save_pretrained(args.pytorch_dump_folder_path)

    if args.push_to_hub:
        print(f"Pushing model for {args.model_name} to hub")
        model.push_to_hub(f"ruffy369/{args.model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model-name",
        default="propainter-hf",
        type=str,
        choices=["propainter-hf"],
        help="Name of the ProPainter model you'd like to convert.",
    )
    parser.add_argument(
        "--optical-flow-checkpoint-url",
        default="https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth",
        type=str,
        help="Url for the optical flow module weights.",
    )
    parser.add_argument(
        "--flow-completion-checkpoint-url",
        default="https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth",
        type=str,
        help="Url for the flow completion module weights.",
    )
    parser.add_argument(
        "--inpaint-generator-checkpoint-url",
        default="https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth",
        type=str,
        help="Url for the inpaint generator module weights.",
    )
    parser.add_argument(
        "--pytorch-dump-folder-path",
        default=None,
        type=str,
        help="Path to the output PyTorch model directory.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Whether or not to push the converted model to the 🤗 hub.",
    )

    args = parser.parse_args()
    convert_propainter_checkpoint(args)
