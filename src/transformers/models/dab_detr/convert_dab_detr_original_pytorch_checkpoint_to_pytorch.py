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
"""Convert DAB-DETR checkpoints."""

import argparse
import gc
import json
import re
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from transformers import ConditionalDetrImageProcessor, DabDetrConfig, DabDetrForObjectDetection
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # convolutional projection + query embeddings + layernorm of decoder + class and bounding box heads
    # for dab-DETR, also convert reference point head and query scale MLP
    r"input_proj\.(bias|weight)": r"input_projection.\1",
    r"refpoint_embed\.weight": r"query_refpoint_embeddings.weight",
    r"class_embed\.(bias|weight)": r"class_embed.\1",
    # negative lookbehind because of the overlap
    r"(?<!transformer\.decoder\.)bbox_embed\.layers\.(\d+)\.(bias|weight)": r"bbox_predictor.layers.\1.\2",
    r"transformer\.encoder\.query_scale\.layers\.(\d+)\.(bias|weight)": r"encoder.query_scale.layers.\1.\2",
    r"transformer\.decoder\.bbox_embed\.layers\.(\d+)\.(bias|weight)": r"decoder.bbox_embed.layers.\1.\2",
    r"transformer\.decoder\.norm\.(bias|weight)": r"decoder.layernorm.\1",
    r"transformer\.decoder\.ref_point_head\.layers\.(\d+)\.(bias|weight)": r"decoder.ref_point_head.layers.\1.\2",
    r"transformer\.decoder\.ref_anchor_head\.layers\.(\d+)\.(bias|weight)": r"decoder.ref_anchor_head.layers.\1.\2",
    r"transformer\.decoder\.query_scale\.layers\.(\d+)\.(bias|weight)": r"decoder.query_scale.layers.\1.\2",
    r"transformer\.decoder\.layers\.0\.ca_qpos_proj\.(bias|weight)": r"decoder.layers.0.layer.1.cross_attn_query_pos_proj.\1",
    # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms + activation function
    # output projection
    r"transformer\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.(bias|weight)": r"encoder.layers.\1.self_attn.out_proj.\2",
    # FFN layers
    r"transformer\.encoder\.layers\.(\d+)\.linear(\d)\.(bias|weight)": r"encoder.layers.\1.fc\2.\3",
    # normalization layers
    # nm1
    r"transformer\.encoder\.layers\.(\d+)\.norm1\.(bias|weight)": r"encoder.layers.\1.self_attn_layer_norm.\2",
    # nm2
    r"transformer\.encoder\.layers\.(\d+)\.norm2\.(bias|weight)": r"encoder.layers.\1.final_layer_norm.\2",
    # activation function weight
    r"transformer\.encoder\.layers\.(\d+)\.activation\.weight": r"encoder.layers.\1.activation_fn.weight",
    #########################################################################################################################################
    # decoder layers: 2 times output projection, 2 feedforward neural networks and 3 layernorms + activiation function weight
    r"transformer\.decoder\.layers\.(\d+)\.self_attn\.out_proj\.(bias|weight)": r"decoder.layers.\1.layer.0.self_attn.output_proj.\2",
    r"transformer\.decoder\.layers\.(\d+)\.cross_attn\.out_proj\.(bias|weight)": r"decoder.layers.\1.layer.1.cross_attn.output_proj.\2",
    # FFNs
    r"transformer\.decoder\.layers\.(\d+)\.linear(\d)\.(bias|weight)": r"decoder.layers.\1.layer.2.fc\2.\3",
    # nm1
    r"transformer\.decoder\.layers\.(\d+)\.norm1\.(bias|weight)": r"decoder.layers.\1.layer.0.self_attn_layer_norm.\2",
    # nm2
    r"transformer\.decoder\.layers\.(\d+)\.norm2\.(bias|weight)": r"decoder.layers.\1.layer.1.cross_attn_layer_norm.\2",
    # nm3
    r"transformer\.decoder\.layers\.(\d+)\.norm3\.(bias|weight)": r"decoder.layers.\1.layer.2.final_layer_norm.\2",
    # activation function weight
    r"transformer\.decoder\.layers\.(\d+)\.activation\.weight": r"decoder.layers.\1.layer.2.activation_fn.weight",
    # q, k, v projections and biases in self-attention in decoder
    r"transformer\.decoder\.layers\.(\d+)\.sa_qcontent_proj\.(bias|weight)": r"decoder.layers.\1.layer.0.self_attn_query_content_proj.\2",
    r"transformer\.decoder\.layers\.(\d+)\.sa_kcontent_proj\.(bias|weight)": r"decoder.layers.\1.layer.0.self_attn_key_content_proj.\2",
    r"transformer\.decoder\.layers\.(\d+)\.sa_qpos_proj\.(bias|weight)": r"decoder.layers.\1.layer.0.self_attn_query_pos_proj.\2",
    r"transformer\.decoder\.layers\.(\d+)\.sa_kpos_proj\.(bias|weight)": r"decoder.layers.\1.layer.0.self_attn_key_pos_proj.\2",
    r"transformer\.decoder\.layers\.(\d+)\.sa_v_proj\.(bias|weight)": r"decoder.layers.\1.layer.0.self_attn_value_proj.\2",
    # q, k, v projections in cross-attention in decoder
    r"transformer\.decoder\.layers\.(\d+)\.ca_qcontent_proj\.(bias|weight)": r"decoder.layers.\1.layer.1.cross_attn_query_content_proj.\2",
    r"transformer\.decoder\.layers\.(\d+)\.ca_kcontent_proj\.(bias|weight)": r"decoder.layers.\1.layer.1.cross_attn_key_content_proj.\2",
    r"transformer\.decoder\.layers\.(\d+)\.ca_kpos_proj\.(bias|weight)": r"decoder.layers.\1.layer.1.cross_attn_key_pos_proj.\2",
    r"transformer\.decoder\.layers\.(\d+)\.ca_v_proj\.(bias|weight)": r"decoder.layers.\1.layer.1.cross_attn_value_proj.\2",
    r"transformer\.decoder\.layers\.(\d+)\.ca_qpos_sine_proj\.(bias|weight)": r"decoder.layers.\1.layer.1.cross_attn_query_pos_sine_proj.\2",
}


# Copied from transformers.models.mllama.convert_mllama_weights_to_hf.convert_old_keys_to_new_keys
def convert_old_keys_to_new_keys(state_dict_keys: dict = None):
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


def write_image_processor(model_name, pytorch_dump_folder_path, push_to_hub):
    logger.info("Converting image processor...")
    format = "coco_detection"
    image_processor = ConditionalDetrImageProcessor(format=format)
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    image_processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        image_processor.push_to_hub(repo_id=model_name, commit_message="Add new image processor")


@torch.no_grad()
def write_model(model_name, pretrained_model_weights_path, pytorch_dump_folder_path, push_to_hub):
    # load modified config. Why? After loading the default config, the backbone kwargs are already set.
    if "dc5" in model_name:
        config = DabDetrConfig(dilation=True)
    else:
        # load default config
        config = DabDetrConfig()
    # set other attributes
    if "dab-detr-resnet-50-dc5" == model_name:
        config.temperature_height = 10
        config.temperature_width = 10
    if "fixxy" in model_name:
        config.random_refpoints_xy = True
    if "pat3" in model_name:
        config.num_patterns = 3
        # only when the number of patterns (num_patterns parameter in config) are more than 0 like r50-pat3 or r50dc5-pat3
        ORIGINAL_TO_CONVERTED_KEY_MAPPING.update({r"transformer.patterns.weight": r"patterns.weight"})

    config.num_labels = 91
    repo_id = "huggingface/label-files"
    filename = "coco-detection-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}
    # load original model from local path
    loaded = torch.load(pretrained_model_weights_path, map_location=torch.device("cpu"))["model"]
    # Renaming the original model state dictionary to HF compatibile
    all_keys = list(loaded.keys())
    new_keys = convert_old_keys_to_new_keys(all_keys)
    state_dict = {}
    for key in all_keys:
        if "backbone.0.body" in key:
            new_key = key.replace("backbone.0.body", "backbone.conv_encoder.model._backbone")
            state_dict[new_key] = loaded[key]
        # Q, K, V encoder values mapping
        elif re.search("self_attn.in_proj_(weight|bias)", key):
            # Dynamically find the layer number
            pattern = r"layers\.(\d+)\.self_attn\.in_proj_(weight|bias)"
            match = re.search(pattern, key)
            if match:
                layer_num = match.group(1)
            else:
                raise ValueError(f"Pattern not found in key: {key}")

            in_proj_value = loaded.pop(key)
            if "weight" in key:
                state_dict[f"encoder.layers.{layer_num}.self_attn.q_proj.weight"] = in_proj_value[:256, :]
                state_dict[f"encoder.layers.{layer_num}.self_attn.k_proj.weight"] = in_proj_value[256:512, :]
                state_dict[f"encoder.layers.{layer_num}.self_attn.v_proj.weight"] = in_proj_value[-256:, :]
            elif "bias" in key:
                state_dict[f"encoder.layers.{layer_num}.self_attn.q_proj.bias"] = in_proj_value[:256]
                state_dict[f"encoder.layers.{layer_num}.self_attn.k_proj.bias"] = in_proj_value[256:512]
                state_dict[f"encoder.layers.{layer_num}.self_attn.v_proj.bias"] = in_proj_value[-256:]
        else:
            new_key = new_keys[key]
            state_dict[new_key] = loaded[key]

    del loaded
    gc.collect()
    # important: we need to prepend a prefix to each of the base model keys as the head models use different attributes for them
    prefix = "model."
    for key in state_dict.copy().keys():
        if not key.startswith("class_embed") and not key.startswith("bbox_predictor"):
            val = state_dict.pop(key)
            state_dict[prefix + key] = val
    # finally, create HuggingFace model and load state dict
    model = DabDetrForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Saving PyTorch model to {pytorch_dump_folder_path}...")
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    model.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        model.push_to_hub(repo_id=model_name, commit_message="Add new model")


def convert_dab_detr_checkpoint(model_name, pretrained_model_weights_path, pytorch_dump_folder_path, push_to_hub):
    logger.info("Converting image processor...")
    write_image_processor(model_name, pytorch_dump_folder_path, push_to_hub)

    logger.info(f"Converting model {model_name}...")
    write_model(model_name, pretrained_model_weights_path, pytorch_dump_folder_path, push_to_hub)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="dab-detr-resnet-50",
        type=str,
        help="Name of the DAB_DETR model you'd like to convert.",
    )
    parser.add_argument(
        "--pretrained_model_weights_path",
        default="modelzoo/R50/checkpoint.pth",
        type=str,
        help="The path of the original model weights like: modelzoo/checkpoint.pth",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default="DAB_DETR", type=str, help="Path to the folder to output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub",
        default=True,
        type=bool,
        help="Whether to upload the converted weights and image processor config to the HuggingFace model profile. Default is set to false.",
    )
    args = parser.parse_args()
    convert_dab_detr_checkpoint(
        args.model_name, args.pretrained_model_weights_path, args.pytorch_dump_folder_path, args.push_to_hub
    )
