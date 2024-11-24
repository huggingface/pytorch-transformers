# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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
"""Convert DepthPro checkpoints from the original repository.

URL: https://huggingface.co/apple/DepthPro/tree/main
"""

import argparse
import json
from pathlib import Path
import re

import requests
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from transformers import BitImageProcessor, Dinov2Config, Dinov2ForImageClassification, Dinov2Model
from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, PILImageResampling
from transformers.utils import logging

# TODO: import directly from transformers
from transformers.models.depth_pro.configuration_depth_pro import DepthProConfig
from transformers.models.depth_pro.modeling_depth_pro import DepthProForDepthEstimation


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def create_vit_rename_keys(config):
    rename_keys = []
    # fmt: off

    # patch embedding layer
    rename_keys.append(("cls_token", "embeddings.cls_token"))
    rename_keys.append(("pos_embed", "embeddings.position_embeddings"))
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

# we split up the matrix of each encoder layer into queries, keys and values
def read_in_q_k_v(state_dict, config):
    state_dict_keys = state_dict.keys()
    for key in list(state_dict_keys):
        if "qkv" in key:
            in_proj = state_dict.pop(key)
            q, k, v = torch.split(in_proj, config.hidden_size, dim=0)

            if "fov" in key:
                key = key.replace('fov.encoder.0', 'fov_model.encoder')
            else:
                key = "depth_pro." + key

            key = key.replace("blocks", "encoder.layer")
            state_dict[key.replace("attn.qkv", "attention.attention.query")] = q
            state_dict[key.replace("attn.qkv", "attention.attention.key")] = k
            state_dict[key.replace("attn.qkv", "attention.attention.value")] = v
    return state_dict

# hard coded upsample keys
def update_hard_coded_keys(state_dict):
    mapping = [
        # upsamples
        ('encoder.upsample_latent0.0.weight', 'depth_pro.encoder.upsample_intermediate.1.proj.weight'),
        ('encoder.upsample_latent0.1.weight', 'depth_pro.encoder.upsample_intermediate.1.upsample_blocks.0.weight'),
        ('encoder.upsample_latent0.2.weight', 'depth_pro.encoder.upsample_intermediate.1.upsample_blocks.1.weight'),
        ('encoder.upsample_latent0.3.weight', 'depth_pro.encoder.upsample_intermediate.1.upsample_blocks.2.weight'),
        ('encoder.upsample_latent1.0.weight', 'depth_pro.encoder.upsample_intermediate.0.proj.weight'),
        ('encoder.upsample_latent1.1.weight', 'depth_pro.encoder.upsample_intermediate.0.upsample_blocks.0.weight'),
        ('encoder.upsample_latent1.2.weight', 'depth_pro.encoder.upsample_intermediate.0.upsample_blocks.1.weight'),
        ('encoder.upsample0.0.weight', 'depth_pro.encoder.upsample_scaled_images.2.proj.weight'),
        ('encoder.upsample0.1.weight', 'depth_pro.encoder.upsample_scaled_images.2.upsample_blocks.0.weight'),
        ('encoder.upsample1.0.weight', 'depth_pro.encoder.upsample_scaled_images.1.proj.weight'),
        ('encoder.upsample1.1.weight', 'depth_pro.encoder.upsample_scaled_images.1.upsample_blocks.0.weight'),
        ('encoder.upsample2.0.weight', 'depth_pro.encoder.upsample_scaled_images.0.proj.weight'),
        ('encoder.upsample2.1.weight', 'depth_pro.encoder.upsample_scaled_images.0.upsample_blocks.0.weight'),
        ('encoder.upsample_lowres.weight', 'depth_pro.encoder.upsample_image.upsample_blocks.0.weight'),
        ('encoder.upsample_lowres.bias', 'depth_pro.encoder.upsample_image.upsample_blocks.0.bias'),

        # neck
        ("fov.downsample.0.weight", "fov_model.global_neck.0.weight"),
        ("fov.downsample.0.bias", "fov_model.global_neck.0.bias"),
        ("fov.encoder.1.weight", "fov_model.encoder_neck.weight"),
        ("fov.encoder.1.bias", "fov_model.encoder_neck.bias"),
    ]
    for src, dest in mapping:
        state_dict[dest] = state_dict.pop(src)
    
    return state_dict



# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    return image



@torch.no_grad()
def convert_depth_pro_checkpoint(repo_id, filename, pytorch_dump_folder_path, push_to_hub=False):
    """
    Copy/paste/tweak model's weights to our DepthPro structure.
    """

    # define default DepthPro configuration
    config = DepthProConfig()

    # load original weights from huggingface hub
    # TODO: download from hub
    # file_path = hf_hub_download(repo_id, filename)
    file_path = "/home/geetu/work/hf/depth_pro/depth_pro.pt"
    state_dict = torch.load(file_path, weights_only=True)

    # enumerate fusion layers
    n_scaled_images = len(config.scaled_images_ratios)       # 3
    n_intermediate_hooks = len(config.intermediate_hook_ids) # 2
    n_fusion_layers = n_scaled_images + n_intermediate_hooks # 5

    # 1. keys for vit encoders
    vit_rename_keys = create_vit_rename_keys(config)
    for src_prefix, dest_prefix in [
        ("encoder.patch_encoder", "depth_pro.encoder.patch_encoder"),
        ("encoder.image_encoder", "depth_pro.encoder.image_encoder"),
        ("fov.encoder.0", "fov_model.encoder"),
    ]:
        for src, dest in vit_rename_keys:
            src = src_prefix + "." + src
            dest = dest_prefix + "." + dest
            state_dict[dest] = state_dict.pop(src)

    # 2. qkv keys for vit encoders
    state_dict = read_in_q_k_v(state_dict, config)

    # 3. hard coded mapping
    state_dict = update_hard_coded_keys(state_dict)


    for key in list(state_dict.keys()):

        # 4. final depth estimation head
        if key.startswith("head."):
            new_key = "head." + key

        # 5. fov model head
        elif key.startswith("fov.head."):
            new_key = key.replace("fov", 'fov_model')

        # 6. projections between encoder and fusion
        elif "decoder.convs." in key:
            n = re.findall(r'\d+', key)[0] # find digit inside string
            n = n_fusion_layers - int(n) - 1
            new_key = f"projections.{n}.weight"

        # 7. fuse low res with image features
        elif "encoder.fuse_lowres." in key:
            new_key = key.replace("encoder.fuse_lowres", "depth_pro.encoder.fuse_image_with_low_res")

        # 8. fusion stage (decoder)
        elif key.startswith("decoder.fusions."):
            new_key = key.replace("decoder.fusions.", "fusion_stage.layers.")
            new_key = new_key.replace("resnet1", "residual_layer1")
            new_key = new_key.replace("resnet2", "residual_layer2")
            new_key = new_key.replace("residual.1", "convolution1")
            new_key = new_key.replace("residual.3", "convolution2")
            new_key = new_key.replace("out_conv", "projection")

            n_with_dots = re.findall(r'.\d+.', new_key)[0] # find digit inside string followed by .
            n = n_with_dots[1:-1]
            n = n_fusion_layers - int(n) - 1
            new_key = new_key.replace(n_with_dots, f".{n}.")

        else:
            continue

        state_dict[new_key] = state_dict.pop(key)        

    model = DepthProForDepthEstimation(config, use_fov_model=True).eval()
    model.load_state_dict(state_dict)

    exit()

    # ----------------

    

    for key, val in state_dict.copy().items():
        val = state_dict.pop(key)
        if "w12" in key:
            key = key.replace("w12", "weights_in")
        if "w3" in key:
            key = key.replace("w3", "weights_out")
        state_dict[key] = val

    # load HuggingFace model
    if image_classifier:
        model = Dinov2ForImageClassification(config).eval()
        model.dinov2.load_state_dict(state_dict)
        model_name_to_classifier_dict_url = {
            "dinov2_vits14_1layer": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_linear_head.pth",
            "dinov2_vitb14_1layer": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_linear_head.pth",
            "dinov2_vitl14_1layer": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_linear_head.pth",
            "dinov2_vitg14_1layer": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_linear_head.pth",
        }
        url = model_name_to_classifier_dict_url[model_name]
        classifier_state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        model.classifier.weight = nn.Parameter(classifier_state_dict["weight"])
        model.classifier.bias = nn.Parameter(classifier_state_dict["bias"])
    else:
        model = Dinov2Model(config).eval()
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
            "dinov2_vits14": "dinov2-small",
            "dinov2_vitb14": "dinov2-base",
            "dinov2_vitl14": "dinov2-large",
            "dinov2_vitg14": "dinov2-giant",
            "dinov2_vits14_1layer": "dinov2-small-imagenet1k-1-layer",
            "dinov2_vitb14_1layer": "dinov2-base-imagenet1k-1-layer",
            "dinov2_vitl14_1layer": "dinov2-large-imagenet1k-1-layer",
            "dinov2_vitg14_1layer": "dinov2-giant-imagenet1k-1-layer",
        }

        name = model_name_to_hf_name[model_name]
        model.push_to_hub(f"facebook/{name}")
        processor.push_to_hub(f"facebook/{name}")


convert_depth_pro_checkpoint("apple/DepthPro", "depth_pro.pt", "yooo_torch_dump", False)
exit()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--repo_id", default="apple/DepthPro", type=str, help="Name of the repo from huggingface you'd like to convert."
    )
    parser.add_argument(
        "--filename", default="depth_pro.pt", type=str, help="Name of the file from repo you'd like to convert."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )

    args = parser.parse_args()
    convert_depth_pro_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub)
