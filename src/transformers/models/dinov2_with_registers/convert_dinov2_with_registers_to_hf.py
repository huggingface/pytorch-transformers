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
"""Convert DINOv2 with Registers checkpoints from the original repository.

URL: https://github.com/facebookresearch/dinov2/tree/main
"""

import argparse
import json
from pathlib import Path

import requests
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from transformers import (
    BitImageProcessor,
    Dinov2WithRegistersConfig,
    Dinov2WithRegistersForImageClassification,
    Dinov2WithRegistersModel,
)
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_dinov2_with_registers_config(model_name, image_classifier=False):
    config = Dinov2WithRegistersConfig(image_size=518, patch_size=14)

    # size of the architecture
    if "vits" in model_name:
        config.hidden_size = 384
        config.num_attention_heads = 6
    elif "vitb" in model_name:
        pass
    elif "vitl" in model_name:
        config.hidden_size = 1024
        config.num_hidden_layers = 24
        config.num_attention_heads = 16
    elif "vitg" in model_name:
        config.use_swiglu_ffn = True
        config.hidden_size = 1536
        config.num_hidden_layers = 40
        config.num_attention_heads = 24
    else:
        raise ValueError("Model not supported")

    if image_classifier:
        repo_id = "huggingface/label-files"
        filename = "imagenet-1k-id2label.json"
        config.num_labels = 1000
        config.id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        config.id2label = {int(k): v for k, v in config.id2label.items()}

    return config


def create_rename_keys(config):
    rename_keys = []
    # fmt: off

    # patch embedding layer
    rename_keys.append(("cls_token", "embeddings.cls_token"))
    rename_keys.append(("mask_token", "embeddings.mask_token"))
    rename_keys.append(("pos_embed", "embeddings.position_embeddings"))
    rename_keys.append(("register_tokens", "embeddings.register_tokens"))
    rename_keys.append(("patch_embed.proj.weight", "embeddings.patch_embeddings.projection.weight"))
    rename_keys.append(("patch_embed.proj.bias", "embeddings.patch_embeddings.projection.bias"))

    for i in range(config.num_hidden_layers):
        # layernorms
        rename_keys.append((f"blocks.{i}.norm1.weight", f"encoder.layer.{i}.norm1.weight"))
        rename_keys.append((f"blocks.{i}.norm1.bias", f"encoder.layer.{i}.norm1.bias"))
        rename_keys.append((f"blocks.{i}.norm2.weight", f"encoder.layer.{i}.norm2.weight"))
        rename_keys.append((f"blocks.{i}.norm2.bias", f"encoder.layer.{i}.norm2.bias"))
        # MLP
        if config.use_swiglu_ffn:
            rename_keys.append((f"blocks.{i}.mlp.w12.weight", f"encoder.layer.{i}.mlp.w12.weight"))
            rename_keys.append((f"blocks.{i}.mlp.w12.bias", f"encoder.layer.{i}.mlp.w12.bias"))
            rename_keys.append((f"blocks.{i}.mlp.w3.weight", f"encoder.layer.{i}.mlp.w3.weight"))
            rename_keys.append((f"blocks.{i}.mlp.w3.bias", f"encoder.layer.{i}.mlp.w3.bias"))
        else:
            rename_keys.append((f"blocks.{i}.mlp.fc1.weight", f"encoder.layer.{i}.mlp.fc1.weight"))
            rename_keys.append((f"blocks.{i}.mlp.fc1.bias", f"encoder.layer.{i}.mlp.fc1.bias"))
            rename_keys.append((f"blocks.{i}.mlp.fc2.weight", f"encoder.layer.{i}.mlp.fc2.weight"))
            rename_keys.append((f"blocks.{i}.mlp.fc2.bias", f"encoder.layer.{i}.mlp.fc2.bias"))
        # layerscale
        rename_keys.append((f"blocks.{i}.ls1.gamma", f"encoder.layer.{i}.layer_scale1.lambda1"))
        rename_keys.append((f"blocks.{i}.ls2.gamma", f"encoder.layer.{i}.layer_scale2.lambda1"))
        # attention projection layer
        rename_keys.append((f"blocks.{i}.attn.proj.weight", f"encoder.layer.{i}.attention.output.dense.weight"))
        rename_keys.append((f"blocks.{i}.attn.proj.bias", f"encoder.layer.{i}.attention.output.dense.bias"))

    # final layernorm
    rename_keys.append(("norm.weight", "layernorm.weight"))
    rename_keys.append(("norm.bias", "layernorm.bias"))

    # fmt: on
    return rename_keys


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


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


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image


@torch.no_grad()
def convert_dinov2_with_registers_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our Dinov2WithRegisters structure.
    """

    # define default Dinov2WithRegisters configuration
    image_classifier = "1layer" in model_name
    config = get_dinov2_with_registers_config(model_name, image_classifier=image_classifier)

    # load original model from torch hub
    original_model = torch.hub.load("facebookresearch/dinov2", model_name.replace("_1layer", ""))
    original_model.eval()

    # load state_dict of original model, remove and rename some keys
    state_dict = original_model.state_dict()
    rename_keys = create_rename_keys(config)
    for src, dest in rename_keys:
        rename_key(state_dict, src, dest)
    read_in_q_k_v(state_dict, config)

    for key, val in state_dict.copy().items():
        val = state_dict.pop(key)
        if "w12" in key:
            key = key.replace("w12", "weights_in")
        if "w3" in key:
            key = key.replace("w3", "weights_out")
        state_dict[key] = val

    # load HuggingFace model
    if image_classifier:
        model = Dinov2WithRegistersForImageClassification(config).eval()
        model.dinov2_with_registers.load_state_dict(state_dict)
        model_name_to_classifier_dict_url = {
            "dinov2_vits14_reg_1layer": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_reg4_linear_head.pth",
            "dinov2_vitb14_reg_1layer": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_linear_head.pth",
            "dinov2_vitl14_reg_1layer": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_linear_head.pth",
            "dinov2_vitg14_reg_1layer": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_linear_head.pth",
        }
        url = model_name_to_classifier_dict_url[model_name]
        classifier_state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.classifier.weight = nn.Parameter(classifier_state_dict["weight"])
        model.classifier.bias = nn.Parameter(classifier_state_dict["bias"])
    else:
        model = Dinov2WithRegistersModel(config).eval()
        model.load_state_dict(state_dict)

    # load image
    image = prepare_img()

    # preprocess image
    transformations = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_DEFAULT_MEAN,  # these are RGB mean+std values
                std=IMAGENET_DEFAULT_STD,  # across a large photo dataset.
            ),
        ]
    )

    original_pixel_values = transformations(image).unsqueeze(0)  # insert batch dimension

    processor = BitImageProcessor(
        size={"shortest_edge": 256},
        resample=PILImageResampling.BICUBIC,
        image_mean=IMAGENET_DEFAULT_MEAN,
        image_std=IMAGENET_DEFAULT_STD,
    )
    pixel_values = processor(image, return_tensors="pt").pixel_values

    assert torch.allclose(original_pixel_values, pixel_values)

    with torch.no_grad():
        outputs = model(pixel_values, output_hidden_states=True)
        original_outputs = original_model(pixel_values)

    # assert values
    if image_classifier:
        print("Predicted class:")
        class_idx = outputs.logits.argmax(-1).item()
        print(model.config.id2label[class_idx])
    else:
        assert outputs.last_hidden_state[:, 0].shape == original_outputs.shape
        assert torch.allclose(outputs.last_hidden_state[:, 0], original_outputs, atol=1e-3)
    print("Looks ok!")

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        print(f"Saving image processor to {pytorch_dump_folder_path}")
        processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        model_name_to_hf_name = {
            "dinov2_vits14_reg": "dinov2-with-registers-small",
            "dinov2_vitb14_reg": "dinov2-with-registers-base",
            "dinov2_vitl14_reg": "dinov2-with-registers-large",
            "dinov2_vitg14_reg": "dinov2-with-registers-giant",
            "dinov2_vits14_reg_1layer": "dinov2-with-registers-small-imagenet1k-1-layer",
            "dinov2_vitb14_reg_1layer": "dinov2-with-registers-base-imagenet1k-1-layer",
            "dinov2_vitl14_reg_1layer": "dinov2-with-registers-large-imagenet1k-1-layer",
            "dinov2_vitg14_reg_1layer": "dinov2-with-registers-giant-imagenet1k-1-layer",
        }

        name = model_name_to_hf_name[model_name]
        model.push_to_hub(f"nielsr/{name}")
        processor.push_to_hub(f"nielsr/{name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--model_name",
        default="dinov2_vits14_reg",
        type=str,
        choices=[
            "dinov2_vits14_reg",
            "dinov2_vitb14_reg",
            "dinov2_vitl14_reg",
            "dinov2_vitg14_reg",
            "dinov2_vits14_reg_1layer",
            "dinov2_vitb14_reg_1layer",
            "dinov2_vitl14_reg_1layer",
            "dinov2_vitg14_reg_1layer",
        ],
        help="Name of the model you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_dinov2_with_registers_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
