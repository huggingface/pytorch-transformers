# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""Convert MaskFormer checkpoints with ResNet backbone from the original repository. URL:
https://github.com/facebookresearch/MaskFormer"""

import argparse
import json
import pickle
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import MaskFormerConfig, MaskFormerForInstanceSegmentation, MaskFormerImageProcessor, ResNetConfig
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_maskformer_config(model_name: str):
    if "resnet101c" in model_name:
        # TODO add support for ResNet-C backbone, which uses a "deeplab" stem
        raise NotImplementedError("To do")
    elif "resnet101" in model_name:
        backbone_config = ResNetConfig.from_pretrained(
            "microsoft/resnet-101", out_features=["stage1", "stage2", "stage3", "stage4"]
        )
    else:
        backbone_config = ResNetConfig.from_pretrained(
            "microsoft/resnet-50", out_features=["stage1", "stage2", "stage3", "stage4"]
        )
    config = MaskFormerConfig(backbone_config=backbone_config)

    repo_id = "huggingface/label-files"
    if "ade20k-full" in model_name:
        config.num_labels = 847
        filename = "maskformer-ade20k-full-id2label.json"
    elif "ade" in model_name:
        config.num_labels = 150
        filename = "ade20k-id2label.json"
    elif "coco-stuff" in model_name:
        config.num_labels = 171
        filename = "maskformer-coco-stuff-id2label.json"
    elif "coco" in model_name:
        # TODO
        config.num_labels = 133
        filename = "coco-panoptic-id2label.json"
    elif "cityscapes" in model_name:
        config.num_labels = 19
        filename = "cityscapes-id2label.json"
    elif "vistas" in model_name:
        config.num_labels = 65
        filename = "mapillary-vistas-id2label.json"

    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    return config


def create_rename_keys(config):
    rename_keys = []
    # stem
    # fmt: off
    rename_keys.append(("backbone.stem.conv1.weight", "model.pixel_level_module.encoder.embedder.embedder.convolution.weight"))
    rename_keys.append(("backbone.stem.conv1.norm.weight", "model.pixel_level_module.encoder.embedder.embedder.normalization.weight"))
    rename_keys.append(("backbone.stem.conv1.norm.bias", "model.pixel_level_module.encoder.embedder.embedder.normalization.bias"))
    rename_keys.append(("backbone.stem.conv1.norm.running_mean", "model.pixel_level_module.encoder.embedder.embedder.normalization.running_mean"))
    rename_keys.append(("backbone.stem.conv1.norm.running_var", "model.pixel_level_module.encoder.embedder.embedder.normalization.running_var"))
    # fmt: on
    # stages
    for stage_idx in range(len(config.backbone_config.depths)):
        for layer_idx in range(config.backbone_config.depths[stage_idx]):
            # shortcut
            if layer_idx == 0:
                rename_keys.append(
                    (
                        f"backbone.res{stage_idx + 2}.{layer_idx}.shortcut.weight",
                        f"model.pixel_level_module.encoder.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.convolution.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.res{stage_idx + 2}.{layer_idx}.shortcut.norm.weight",
                        f"model.pixel_level_module.encoder.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.res{stage_idx + 2}.{layer_idx}.shortcut.norm.bias",
                        f"model.pixel_level_module.encoder.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.bias",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.res{stage_idx + 2}.{layer_idx}.shortcut.norm.running_mean",
                        f"model.pixel_level_module.encoder.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.running_mean",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.res{stage_idx + 2}.{layer_idx}.shortcut.norm.running_var",
                        f"model.pixel_level_module.encoder.encoder.stages.{stage_idx}.layers.{layer_idx}.shortcut.normalization.running_var",
                    )
                )
            # 3 convs
            for i in range(3):
                rename_keys.append(
                    (
                        f"backbone.res{stage_idx + 2}.{layer_idx}.conv{i+1}.weight",
                        f"model.pixel_level_module.encoder.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.convolution.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.res{stage_idx + 2}.{layer_idx}.conv{i+1}.norm.weight",
                        f"model.pixel_level_module.encoder.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.normalization.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.res{stage_idx + 2}.{layer_idx}.conv{i+1}.norm.bias",
                        f"model.pixel_level_module.encoder.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.normalization.bias",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.res{stage_idx + 2}.{layer_idx}.conv{i+1}.norm.running_mean",
                        f"model.pixel_level_module.encoder.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.normalization.running_mean",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.res{stage_idx + 2}.{layer_idx}.conv{i+1}.norm.running_var",
                        f"model.pixel_level_module.encoder.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.{i}.normalization.running_var",
                    )
                )

    # FPN
    # fmt: off
    rename_keys.append(("sem_seg_head.layer_4.weight", "model.pixel_level_module.decoder.fpn.stem.0.weight"))
    rename_keys.append(("sem_seg_head.layer_4.norm.weight", "model.pixel_level_module.decoder.fpn.stem.1.weight"))
    rename_keys.append(("sem_seg_head.layer_4.norm.bias", "model.pixel_level_module.decoder.fpn.stem.1.bias"))
    for source_index, target_index in zip(range(3, 0, -1), range(0, 3)):
        rename_keys.append((f"sem_seg_head.adapter_{source_index}.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.proj.0.weight"))
        rename_keys.append((f"sem_seg_head.adapter_{source_index}.norm.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.proj.1.weight"))
        rename_keys.append((f"sem_seg_head.adapter_{source_index}.norm.bias", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.proj.1.bias"))
        rename_keys.append((f"sem_seg_head.layer_{source_index}.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.block.0.weight"))
        rename_keys.append((f"sem_seg_head.layer_{source_index}.norm.weight", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.block.1.weight"))
        rename_keys.append((f"sem_seg_head.layer_{source_index}.norm.bias", f"model.pixel_level_module.decoder.fpn.layers.{target_index}.block.1.bias"))
    rename_keys.append(("sem_seg_head.mask_features.weight", "model.pixel_level_module.decoder.mask_projection.weight"))
    rename_keys.append(("sem_seg_head.mask_features.bias", "model.pixel_level_module.decoder.mask_projection.bias"))
    # fmt: on

    # Transformer decoder
    # fmt: off
    for idx in range(config.decoder_config.decoder_layers):
        # self-attention out projection
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.self_attn.out_proj.weight", f"model.transformer_module.decoder.layers.{idx}.self_attn.out_proj.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.self_attn.out_proj.bias", f"model.transformer_module.decoder.layers.{idx}.self_attn.out_proj.bias"))
        # cross-attention out projection
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.multihead_attn.out_proj.weight", f"model.transformer_module.decoder.layers.{idx}.encoder_attn.out_proj.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.multihead_attn.out_proj.bias", f"model.transformer_module.decoder.layers.{idx}.encoder_attn.out_proj.bias"))
        # MLP 1
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear1.weight", f"model.transformer_module.decoder.layers.{idx}.fc1.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear1.bias", f"model.transformer_module.decoder.layers.{idx}.fc1.bias"))
        # MLP 2
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear2.weight", f"model.transformer_module.decoder.layers.{idx}.fc2.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.linear2.bias", f"model.transformer_module.decoder.layers.{idx}.fc2.bias"))
        # layernorm 1 (self-attention layernorm)
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm1.weight", f"model.transformer_module.decoder.layers.{idx}.self_attn_layer_norm.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm1.bias", f"model.transformer_module.decoder.layers.{idx}.self_attn_layer_norm.bias"))
        # layernorm 2 (cross-attention layernorm)
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm2.weight", f"model.transformer_module.decoder.layers.{idx}.encoder_attn_layer_norm.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm2.bias", f"model.transformer_module.decoder.layers.{idx}.encoder_attn_layer_norm.bias"))
        # layernorm 3 (final layernorm)
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm3.weight", f"model.transformer_module.decoder.layers.{idx}.final_layer_norm.weight"))
        rename_keys.append((f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.norm3.bias", f"model.transformer_module.decoder.layers.{idx}.final_layer_norm.bias"))

    rename_keys.append(("sem_seg_head.predictor.transformer.decoder.norm.weight", "model.transformer_module.decoder.layernorm.weight"))
    rename_keys.append(("sem_seg_head.predictor.transformer.decoder.norm.bias", "model.transformer_module.decoder.layernorm.bias"))
    # fmt: on

    # heads on top
    # fmt: off
    rename_keys.append(("sem_seg_head.predictor.query_embed.weight", "model.transformer_module.queries_embedder.weight"))

    rename_keys.append(("sem_seg_head.predictor.input_proj.weight", "model.transformer_module.input_projection.weight"))
    rename_keys.append(("sem_seg_head.predictor.input_proj.bias", "model.transformer_module.input_projection.bias"))

    rename_keys.append(("sem_seg_head.predictor.class_embed.weight", "class_predictor.weight"))
    rename_keys.append(("sem_seg_head.predictor.class_embed.bias", "class_predictor.bias"))

    for i in range(3):
        rename_keys.append((f"sem_seg_head.predictor.mask_embed.layers.{i}.weight", f"mask_embedder.{i}.0.weight"))
        rename_keys.append((f"sem_seg_head.predictor.mask_embed.layers.{i}.bias", f"mask_embedder.{i}.0.bias"))
    # fmt: on

    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


# we split up the matrix of each encoder layer into queries, keys and values
def read_in_decoder_q_k_v(state_dict, config):
    # fmt: off
    hidden_size = config.decoder_config.hidden_size
    for idx in range(config.decoder_config.decoder_layers):
        # read in weights + bias of self-attention input projection layer (in the original implementation, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.q_proj.weight"] = in_proj_weight[: hidden_size, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.q_proj.bias"] = in_proj_bias[:config.hidden_size]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.k_proj.weight"] = in_proj_weight[hidden_size : hidden_size * 2, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.k_proj.bias"] = in_proj_bias[hidden_size : hidden_size * 2]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.v_proj.weight"] = in_proj_weight[-hidden_size :, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.self_attn.v_proj.bias"] = in_proj_bias[-hidden_size :]
        # read in weights + bias of cross-attention input projection layer (in the original implementation, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.multihead_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"sem_seg_head.predictor.transformer.decoder.layers.{idx}.multihead_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.q_proj.weight"] = in_proj_weight[: hidden_size, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.q_proj.bias"] = in_proj_bias[:config.hidden_size]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.k_proj.weight"] = in_proj_weight[hidden_size : hidden_size * 2, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.k_proj.bias"] = in_proj_bias[hidden_size : hidden_size * 2]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.v_proj.weight"] = in_proj_weight[-hidden_size :, :]
        state_dict[f"model.transformer_module.decoder.layers.{idx}.encoder_attn.v_proj.bias"] = in_proj_bias[-hidden_size :]
    # fmt: on


# We will verify our results on an image of cute cats
def prepare_img() -> torch.Tensor:
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


@torch.no_grad()
def convert_maskformer_checkpoint(
    model_name: str, checkpoint_path: str, pytorch_dump_folder_path: str, push_to_hub: bool = False
):
    """
    Copy/paste/tweak model's weights to our MaskFormer structure.
    """
    config = get_maskformer_config(model_name)

    # load original state_dict
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    state_dict = data["model"]

    # rename keys
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_decoder_q_k_v(state_dict, config)

    # update to torch tensors
    for key, value in state_dict.items():
        state_dict[key] = torch.from_numpy(value)

    # load 🤗 model
    model = MaskFormerForInstanceSegmentation(config)
    model.eval()

    model.load_state_dict(state_dict)

    # verify results
    image = prepare_img()
    if "vistas" in model_name:
        ignore_index = 65
    elif "cityscapes" in model_name:
        ignore_index = 65535
    else:
        ignore_index = 255
    do_reduce_labels = True if "ade" in model_name else False
    image_processor = MaskFormerImageProcessor(ignore_index=ignore_index, do_reduce_labels=do_reduce_labels)

    inputs = image_processor(image, return_tensors="pt")

    outputs = model(**inputs)

    if model_name == "maskformer-resnet50-ade":
        expected_logits = torch.tensor(
            [[6.7710, -0.1452, -3.5687], [1.9165, -1.0010, -1.8614], [3.6209, -0.2950, -1.3813]]
        )
    elif model_name == "maskformer-resnet101-ade":
        expected_logits = torch.tensor(
            [[4.0381, -1.1483, -1.9688], [2.7083, -1.9147, -2.2555], [3.4367, -1.3711, -2.1609]]
        )
    elif model_name == "maskformer-resnet50-coco-stuff":
        expected_logits = torch.tensor(
            [[3.2309, -3.0481, -2.8695], [5.4986, -5.4242, -2.4211], [6.2100, -5.2279, -2.7786]]
        )
    elif model_name == "maskformer-resnet101-coco-stuff":
        expected_logits = torch.tensor(
            [[4.7188, -3.2585, -2.8857], [6.6871, -2.9181, -1.2487], [7.2449, -2.2764, -2.1874]]
        )
    elif model_name == "maskformer-resnet101-cityscapes":
        expected_logits = torch.tensor(
            [[-1.8861, -1.5465, 0.6749], [-2.3677, -1.6707, -0.0867], [-2.2314, -1.9530, -0.9132]]
        )
    elif model_name == "maskformer-resnet50-vistas":
        expected_logits = torch.tensor(
            [[-6.3917, -1.5216, -1.1392], [-5.5335, -4.5318, -1.8339], [-4.3576, -4.0301, 0.2162]]
        )
    elif model_name == "maskformer-resnet50-ade20k-full":
        expected_logits = torch.tensor(
            [[3.6146, -1.9367, -3.2534], [4.0099, 0.2027, -2.7576], [3.3913, -2.3644, -3.9519]]
        )
    elif model_name == "maskformer-resnet101-ade20k-full":
        expected_logits = torch.tensor(
            [[3.2211, -1.6550, -2.7605], [2.8559, -2.4512, -2.9574], [2.6331, -2.6775, -2.1844]]
        )

    assert torch.allclose(outputs.class_queries_logits[0, :3, :3], expected_logits, atol=1e-4)
    print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        print(f"Saving model and image processor of {model_name} to {pytorch_dump_folder_path}")
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        model.save_pretrained(pytorch_dump_folder_path)
        image_processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        print(f"Pushing model and image processor of {model_name} to the hub...")
        model.push_to_hub(f"facebook/{model_name}")
        image_processor.push_to_hub(f"facebook/{model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="maskformer-resnet50-ade",
        type=str,
        required=True,
        choices=[
            "maskformer-resnet50-ade",
            "maskformer-resnet101-ade",
            "maskformer-resnet50-coco-stuff",
            "maskformer-resnet101-coco-stuff",
            "maskformer-resnet101-cityscapes",
            "maskformer-resnet50-vistas",
            "maskformer-resnet50-ade20k-full",
            "maskformer-resnet101-ade20k-full",
        ],
        help=("Name of the MaskFormer model you'd like to convert",),
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the original pickle file (.pkl) of the original checkpoint.\n"
        "Given the files are in the pickle format, please be wary of passing it files you trust.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_maskformer_checkpoint(
        args.model_name, args.checkpoint_path, args.pytorch_dump_folder_path, args.push_to_hub
    )
