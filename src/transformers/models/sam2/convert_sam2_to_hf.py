# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""
Convert SAM checkpoints from the original repository.

URL: https://github.com/facebookresearch/segment-anything-2.
"""

import argparse
import re

import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    Sam2Config,
    Sam2ImageEncoderConfig,
    Sam2ImageProcessor,
    Sam2MaskDecoderConfig,
    Sam2MemoryAttentionConfig,
    Sam2MemoryEncoderConfig,
    Sam2Model,
    Sam2Processor,
    Sam2PromptEncoderConfig,
)


def get_config(model_name):
    if "sam2.1_hiera_tiny" in model_name:
        image_encoder_config = Sam2ImageEncoderConfig()
        prompt_encoder_config = Sam2PromptEncoderConfig()
        mask_decoder_config = Sam2MaskDecoderConfig()
        memory_attention_config = Sam2MemoryAttentionConfig()
        memory_encoder_config = Sam2MemoryEncoderConfig()
    elif "sam2.1_hiera_small" in model_name:
        # TO DO
        pass
    elif "sam2.1_hiera_base_plus" in model_name:
        # TO DO
        pass
    elif "sam2.1_hiera_large" in model_name:
        # TO DO
        pass

    config = Sam2Config(
        image_encoder_config=image_encoder_config,
        prompt_encoder_config=prompt_encoder_config,
        mask_decoder_config=mask_decoder_config,
        memory_attention_config=memory_attention_config,
        memory_encoder_config=memory_encoder_config,
    )

    return config


KEYS_TO_MODIFY_MAPPING = {
    "iou_prediction_head.layers.0": "iou_prediction_head.proj_in",
    "iou_prediction_head.layers.1": "iou_prediction_head.layers.0",
    "iou_prediction_head.layers.2": "iou_prediction_head.proj_out",
    "mask_decoder.output_upscaling.0": "mask_decoder.upscale_conv1",
    "mask_decoder.output_upscaling.1": "mask_decoder.upscale_layer_norm",
    "mask_decoder.output_upscaling.3": "mask_decoder.upscale_conv2",
    "mask_downscaling.0": "mask_embed.conv1",
    "mask_downscaling.1": "mask_embed.layer_norm1",
    "mask_downscaling.3": "mask_embed.conv2",
    "mask_downscaling.4": "mask_embed.layer_norm2",
    "mask_downscaling.6": "mask_embed.conv3",
    "dwconv": "depthwise_conv",
    "pwconv": "pointwise_conv",
    "fuser": "memory_fuser",
    "point_embeddings": "point_embed",
    "pe_layer.positional_encoding_gaussian_matrix": "shared_embedding.positional_embedding",
    "vision_encoder": "image_encoder",
    "sam_prompt_encoder": "prompt_encoder",
    "sam_mask_decoder": "mask_decoder",
    "maskmem_tpos_enc": "memory_temporal_positional_encoding",
    "gamma": "scale",
    "neck.0": "neck.conv1",
    "neck.1": "neck.layer_norm1",
    "neck.2": "neck.conv2",
    "neck.3": "neck.layer_norm2",
    "pix_feat_proj": "feature_projection",
    "patch_embed.proj": "patch_embed.projection",
    "no_mem_embed": "no_memory_embedding",
    "no_mem_pos_enc": "no_memory_positional_encoding",
    "obj_ptr": "object_pointer",
    ".norm": ".layer_norm",
    "trunk.": "",
}


def replace_keys(state_dict):
    model_state_dict = {}
    output_hypernetworks_mlps_pattern = r".*.output_hypernetworks_mlps.(\d+).layers.(\d+).*"
    output_mask_decoder_mlps_pattern = r"mask_decoder.transformer.layers.(\d+).mlp.layers.(\d+).*"
    output_mask_decoder_score_head_pattern = r"mask_decoder.pred_obj_score_head.layers.(\d+).*"
    output_image_encoder_mlps_pattern = r"image_encoder.blocks.(\d+).mlp.layers.(\d+).*"
    output_image_encoder_neck_pattern = r"image_encoder.neck.convs.(\d+).conv"
    output_memory_encoder_projection_pattern = r"memory_encoder.out_proj.*"
    output_object_pointer_proj_pattern = r"object_pointer_proj.layers.(\d+).*"

    for key, value in state_dict.items():
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        # image_encoder.blocks.0.mlp.layers.1.weight -> image_encoder.blocks.0.mlp.proj_out.weight
        if re.match(output_image_encoder_mlps_pattern, key):
            layer_nb = int(re.match(output_image_encoder_mlps_pattern, key).group(2))
            if layer_nb == 0:
                key = key.replace("layers.0", "proj_in")
            elif layer_nb == 1:
                key = key.replace("layers.1", "proj_out")

        # mask_decoder.transformer.layers.0.mlp.layers.1.weight -> mask_decoder.transformer.layers.1.mlp.proj_out.weight
        if re.match(output_mask_decoder_mlps_pattern, key):
            layer_nb = int(re.match(output_mask_decoder_mlps_pattern, key).group(2))
            if layer_nb == 0:
                key = key.replace("mlp.layers.0", "mlp.proj_in")
            elif layer_nb == 1:
                key = key.replace("mlp.layers.1", "mlp.proj_out")

        # mask_decoder.pred_obj_score_head.layers.1.weight -> mask_decoder.pred_obj_score_head.proj_in.weight
        if re.match(output_mask_decoder_score_head_pattern, key):
            layer_nb = int(re.match(output_mask_decoder_score_head_pattern, key).group(1))
            if layer_nb == 0:
                key = key.replace("layers.0", "proj_in")
            elif layer_nb == 1:
                key = key.replace("layers.1", "layers.0")
            elif layer_nb == 2:
                key = key.replace("layers.2", "proj_out")

        if re.match(output_hypernetworks_mlps_pattern, key):
            layer_nb = int(re.match(output_hypernetworks_mlps_pattern, key).group(2))
            if layer_nb == 0:
                key = key.replace("layers.0", "proj_in")
            elif layer_nb == 1:
                key = key.replace("layers.1", "layers.0")
            elif layer_nb == 2:
                key = key.replace("layers.2", "proj_out")

        # image_encoder.neck.convs.1.conv.bias -> image_encoder.neck.convs.1.bias
        if re.match(output_image_encoder_neck_pattern, key):
            key = key.replace(".conv.", ".")

        # memory_encoder.out_proj.weight -> memory_encoder.projection.weight
        if re.match(output_memory_encoder_projection_pattern, key):
            key = key.replace(".out_proj.", ".projection.")

        if re.match(output_object_pointer_proj_pattern, key):
            layer_nb = int(re.match(output_object_pointer_proj_pattern, key).group(1))
            if layer_nb == 0:
                key = key.replace("layers.0", "proj_in")
            elif layer_nb == 1:
                key = key.replace("layers.1", "layers.0")
            elif layer_nb == 2:
                key = key.replace("layers.2", "proj_out")

        model_state_dict[key] = value

    model_state_dict["shared_image_embedding.positional_embedding"] = model_state_dict[
        "prompt_encoder.shared_embedding.positional_embedding"
    ]

    return model_state_dict


def convert_sam2_checkpoint(model_name, checkpoint_path, pytorch_dump_folder, push_to_hub):
    config = get_config(model_name)

    state_dict = torch.load(checkpoint_path, map_location="cpu")["model"]
    state_dict = replace_keys(state_dict)

    image_processor = Sam2ImageProcessor()
    processor = Sam2Processor(image_processor=image_processor)
    hf_model = Sam2Model(config)
    hf_model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    hf_model.load_state_dict(state_dict, strict=True)
    hf_model = hf_model.to(device)

    img_url = "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

    input_points = [[[1000, 600]]]
    input_labels = [[1]]

    if model_name == "sam2.1_hiera_tiny":
        inputs = processor(
            images=np.array(raw_image), input_points=input_points, input_labels=input_labels, return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = hf_model(**inputs)
        scores = output.iou_scores.squeeze()

        assert torch.allclose(scores, torch.tensor([0.0314, 0.9649, 0.1026]).cuda(), atol=1e-4)

    elif model_name == "sam2_hiera_small":
        # TO DO
        pass
    elif model_name == "sam2_hiera_base_plus":
        # TO DO
        pass

    elif model_name == "sam2_hiera_large":
        # TO DO
        pass

    if pytorch_dump_folder is not None:
        processor.save_pretrained(pytorch_dump_folder)
        hf_model.save_pretrained(pytorch_dump_folder)

    if push_to_hub:
        repo_id = f"danelcsb/{model_name}"
        processor.push_to_hub(repo_id)
        hf_model.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    choices = ["sam2.1_hiera_tiny", "sam2.1_hiera_small", "sam2.1_hiera_base_plus", "sam2.1_hiera_large"]
    parser.add_argument(
        "--model_name",
        default="sam2.1_hiera_tiny",
        choices=choices,
        type=str,
        help="Name of the original model to convert",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        help="Path to the original checkpoint",
    )
    parser.add_argument("--pytorch_dump_folder_path", default="", type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether to push the model and processor to the hub after converting",
    )

    args = parser.parse_args()

    hf_model_name = args.model_name.replace("_", "-")
    checkpoint_path = hf_hub_download(f"facebook/{hf_model_name}", f"{args.model_name}.pt")

    convert_sam2_checkpoint(args.model_name, checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub)
