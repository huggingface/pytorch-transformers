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
"""Convert RT Detr checkpoints with Timm backbone"""

import argparse
import json
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from transformers import RTDetrImageProcessor, RtDetrV2Config, RtDetrV2ForObjectDetection
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_rt_detr_v2_config(model_name: str) -> RtDetrV2Config:
    config = RtDetrV2Config()

    config.num_labels = 80
    repo_id = "huggingface/label-files"
    filename = "coco-detection-mmdet-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    if model_name == "rtdetr_v2_r18vd":
        config.backbone_config.hidden_sizes = [64, 128, 256, 512]
        config.backbone_config.depths = [2, 2, 2, 2]
        config.backbone_config.layer_type = "basic"
        config.encoder_in_channels = [128, 256, 512]
        config.hidden_expansion = 0.5
        config.decoder_layers = 3
    elif model_name == "rtdetr_v2_r34vd":
        config.backbone_config.hidden_sizes = [64, 128, 256, 512]
        config.backbone_config.depths = [3, 4, 6, 3]
        config.backbone_config.layer_type = "basic"
        config.encoder_in_channels = [128, 256, 512]
        config.hidden_expansion = 0.5
        config.decoder_layers = 4
    elif model_name == "rtdetr_v2_r50vd_m":
        config.hidden_expansion = 0.5
    elif model_name == "rtdetr_v2_r50vd":
        pass
    elif model_name == "rtdetr_v2_r101vd":
        config.backbone_config.depths = [3, 4, 23, 3]
        config.encoder_ffn_dim = 2048
        config.encoder_hidden_dim = 384
        config.decoder_in_channels = [384, 384, 384]

    return config


def create_rename_keys(config):
    # here we list all keys to be renamed (original name on the left, our name on the right)
    rename_keys = []

    # stem
    # fmt: off
    last_key = ["weight", "bias", "running_mean", "running_var"]

    for level in range(3):
        rename_keys.append((f"backbone.conv1.conv1_{level+1}.conv.weight", f"model.backbone.model.embedder.embedder.{level}.convolution.weight"))
        for last in last_key:
            rename_keys.append((f"backbone.conv1.conv1_{level+1}.norm.{last}", f"model.backbone.model.embedder.embedder.{level}.normalization.{last}"))

    for stage_idx in range(len(config.backbone_config.depths)):
        for layer_idx in range(config.backbone_config.depths[stage_idx]):
            # shortcut
            if layer_idx == 0:
                if stage_idx == 0:
                    rename_keys.append(
                        (
                            f"backbone.res_layers.{stage_idx}.blocks.0.short.conv.weight",
                            f"model.backbone.model.encoder.stages.{stage_idx}.layers.0.shortcut.convolution.weight",
                        )
                    )
                    for last in last_key:
                        rename_keys.append(
                            (
                                f"backbone.res_layers.{stage_idx}.blocks.0.short.norm.{last}",
                                f"model.backbone.model.encoder.stages.{stage_idx}.layers.0.shortcut.normalization.{last}",
                            )
                        )
                else:
                    rename_keys.append(
                        (
                            f"backbone.res_layers.{stage_idx}.blocks.0.short.conv.conv.weight",
                            f"model.backbone.model.encoder.stages.{stage_idx}.layers.0.shortcut.1.convolution.weight",
                        )
                    )
                    for last in last_key:
                        rename_keys.append(
                            (
                                f"backbone.res_layers.{stage_idx}.blocks.0.short.conv.norm.{last}",
                                f"model.backbone.model.encoder.stages.{stage_idx}.layers.0.shortcut.1.normalization.{last}",
                            )
                        )

            rename_keys.append(
                (
                    f"backbone.res_layers.{stage_idx}.blocks.{layer_idx}.branch2a.conv.weight",
                    f"model.backbone.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.0.convolution.weight",
                )
            )
            for last in last_key:
                rename_keys.append((
                    f"backbone.res_layers.{stage_idx}.blocks.{layer_idx}.branch2a.norm.{last}",
                    f"model.backbone.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.0.normalization.{last}",
                    ))

            rename_keys.append(
                (
                    f"backbone.res_layers.{stage_idx}.blocks.{layer_idx}.branch2b.conv.weight",
                    f"model.backbone.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.1.convolution.weight",
                )
            )
            for last in last_key:
                rename_keys.append((
                    f"backbone.res_layers.{stage_idx}.blocks.{layer_idx}.branch2b.norm.{last}",
                    f"model.backbone.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.1.normalization.{last}",
                    ))

            # https://github.com/lyuwenyu/RT-DETR/blob/94f5e16708329d2f2716426868ec89aa774af016/rtdetr_pytorch/src/nn/backbone/presnet.py#L171
            if config.backbone_config.layer_type != "basic":
                rename_keys.append(
                    (
                        f"backbone.res_layers.{stage_idx}.blocks.{layer_idx}.branch2c.conv.weight",
                        f"model.backbone.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.2.convolution.weight",
                    )
                )
                for last in last_key:
                    rename_keys.append((
                        f"backbone.res_layers.{stage_idx}.blocks.{layer_idx}.branch2c.norm.{last}",
                        f"model.backbone.model.encoder.stages.{stage_idx}.layers.{layer_idx}.layer.2.normalization.{last}",
                        ))
    # fmt: on

    for i in range(config.encoder_layers):
        # encoder layers: output projection, 2 feedforward neural networks and 2 layernorms
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.self_attn.out_proj.weight",
                f"model.encoder.encoder.{i}.layers.0.self_attn.out_proj.weight",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.self_attn.out_proj.bias",
                f"model.encoder.encoder.{i}.layers.0.self_attn.out_proj.bias",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.linear1.weight",
                f"model.encoder.encoder.{i}.layers.0.fc1.weight",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.linear1.bias",
                f"model.encoder.encoder.{i}.layers.0.fc1.bias",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.linear2.weight",
                f"model.encoder.encoder.{i}.layers.0.fc2.weight",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.linear2.bias",
                f"model.encoder.encoder.{i}.layers.0.fc2.bias",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.norm1.weight",
                f"model.encoder.encoder.{i}.layers.0.self_attn_layer_norm.weight",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.norm1.bias",
                f"model.encoder.encoder.{i}.layers.0.self_attn_layer_norm.bias",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.norm2.weight",
                f"model.encoder.encoder.{i}.layers.0.final_layer_norm.weight",
            )
        )
        rename_keys.append(
            (
                f"encoder.encoder.{i}.layers.0.norm2.bias",
                f"model.encoder.encoder.{i}.layers.0.final_layer_norm.bias",
            )
        )

    for j in range(0, 3):
        rename_keys.append((f"encoder.input_proj.{j}.conv.weight", f"model.encoder_input_proj.{j}.0.weight"))
        for last in last_key:
            rename_keys.append((f"encoder.input_proj.{j}.norm.{last}", f"model.encoder_input_proj.{j}.1.{last}"))

    block_levels = 3 if config.backbone_config.layer_type != "basic" else 4

    for i in range(len(config.encoder_in_channels) - 1):
        # encoder layers: hybridencoder parts
        for j in range(1, block_levels):
            rename_keys.append(
                (f"encoder.fpn_blocks.{i}.conv{j}.conv.weight", f"model.encoder.fpn_blocks.{i}.conv{j}.conv.weight")
            )
            for last in last_key:
                rename_keys.append(
                    (
                        f"encoder.fpn_blocks.{i}.conv{j}.norm.{last}",
                        f"model.encoder.fpn_blocks.{i}.conv{j}.norm.{last}",
                    )
                )

        rename_keys.append((f"encoder.lateral_convs.{i}.conv.weight", f"model.encoder.lateral_convs.{i}.conv.weight"))
        for last in last_key:
            rename_keys.append(
                (f"encoder.lateral_convs.{i}.norm.{last}", f"model.encoder.lateral_convs.{i}.norm.{last}")
            )

        for j in range(3):
            for k in range(1, 3):
                rename_keys.append(
                    (
                        f"encoder.fpn_blocks.{i}.bottlenecks.{j}.conv{k}.conv.weight",
                        f"model.encoder.fpn_blocks.{i}.bottlenecks.{j}.conv{k}.conv.weight",
                    )
                )
                for last in last_key:
                    rename_keys.append(
                        (
                            f"encoder.fpn_blocks.{i}.bottlenecks.{j}.conv{k}.norm.{last}",
                            f"model.encoder.fpn_blocks.{i}.bottlenecks.{j}.conv{k}.norm.{last}",
                        )
                    )

        for j in range(1, block_levels):
            rename_keys.append(
                (f"encoder.pan_blocks.{i}.conv{j}.conv.weight", f"model.encoder.pan_blocks.{i}.conv{j}.conv.weight")
            )
            for last in last_key:
                rename_keys.append(
                    (
                        f"encoder.pan_blocks.{i}.conv{j}.norm.{last}",
                        f"model.encoder.pan_blocks.{i}.conv{j}.norm.{last}",
                    )
                )

        for j in range(3):
            for k in range(1, 3):
                rename_keys.append(
                    (
                        f"encoder.pan_blocks.{i}.bottlenecks.{j}.conv{k}.conv.weight",
                        f"model.encoder.pan_blocks.{i}.bottlenecks.{j}.conv{k}.conv.weight",
                    )
                )
                for last in last_key:
                    rename_keys.append(
                        (
                            f"encoder.pan_blocks.{i}.bottlenecks.{j}.conv{k}.norm.{last}",
                            f"model.encoder.pan_blocks.{i}.bottlenecks.{j}.conv{k}.norm.{last}",
                        )
                    )

        rename_keys.append(
            (f"encoder.downsample_convs.{i}.conv.weight", f"model.encoder.downsample_convs.{i}.conv.weight")
        )
        for last in last_key:
            rename_keys.append(
                (f"encoder.downsample_convs.{i}.norm.{last}", f"model.encoder.downsample_convs.{i}.norm.{last}")
            )

    for i in range(config.decoder_layers):
        # decoder layers: 2 times output projection, 2 feedforward neural networks and 3 layernorms
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.self_attn.out_proj.weight",
                f"model.decoder.layers.{i}.self_attn.out_proj.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.self_attn.out_proj.bias",
                f"model.decoder.layers.{i}.self_attn.out_proj.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.sampling_offsets.weight",
                f"model.decoder.layers.{i}.encoder_attn.sampling_offsets.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.sampling_offsets.bias",
                f"model.decoder.layers.{i}.encoder_attn.sampling_offsets.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.attention_weights.weight",
                f"model.decoder.layers.{i}.encoder_attn.attention_weights.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.attention_weights.bias",
                f"model.decoder.layers.{i}.encoder_attn.attention_weights.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.value_proj.weight",
                f"model.decoder.layers.{i}.encoder_attn.value_proj.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.value_proj.bias",
                f"model.decoder.layers.{i}.encoder_attn.value_proj.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.output_proj.weight",
                f"model.decoder.layers.{i}.encoder_attn.output_proj.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.output_proj.bias",
                f"model.decoder.layers.{i}.encoder_attn.output_proj.bias",
            )
        )
        rename_keys.append(
            (f"decoder.decoder.layers.{i}.norm1.weight", f"model.decoder.layers.{i}.self_attn_layer_norm.weight")
        )
        rename_keys.append(
            (f"decoder.decoder.layers.{i}.norm1.bias", f"model.decoder.layers.{i}.self_attn_layer_norm.bias")
        )
        rename_keys.append(
            (f"decoder.decoder.layers.{i}.norm2.weight", f"model.decoder.layers.{i}.encoder_attn_layer_norm.weight")
        )
        rename_keys.append(
            (f"decoder.decoder.layers.{i}.norm2.bias", f"model.decoder.layers.{i}.encoder_attn_layer_norm.bias")
        )
        rename_keys.append((f"decoder.decoder.layers.{i}.linear1.weight", f"model.decoder.layers.{i}.fc1.weight"))
        rename_keys.append((f"decoder.decoder.layers.{i}.linear1.bias", f"model.decoder.layers.{i}.fc1.bias"))
        rename_keys.append((f"decoder.decoder.layers.{i}.linear2.weight", f"model.decoder.layers.{i}.fc2.weight"))
        rename_keys.append((f"decoder.decoder.layers.{i}.linear2.bias", f"model.decoder.layers.{i}.fc2.bias"))
        rename_keys.append(
            (f"decoder.decoder.layers.{i}.norm3.weight", f"model.decoder.layers.{i}.final_layer_norm.weight")
        )
        rename_keys.append(
            (f"decoder.decoder.layers.{i}.norm3.bias", f"model.decoder.layers.{i}.final_layer_norm.bias")
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.cross_attn.num_points_scale",
                f"model.decoder.layers.{i}.encoder_attn.n_points_scale",
            )
        )

    for i in range(config.decoder_layers):
        # decoder + class and bounding box heads
        rename_keys.append(
            (
                f"decoder.dec_score_head.{i}.weight",
                f"model.decoder.class_embed.{i}.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_score_head.{i}.bias",
                f"model.decoder.class_embed.{i}.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_bbox_head.{i}.layers.0.weight",
                f"model.decoder.bbox_embed.{i}.layers.0.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_bbox_head.{i}.layers.0.bias",
                f"model.decoder.bbox_embed.{i}.layers.0.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_bbox_head.{i}.layers.1.weight",
                f"model.decoder.bbox_embed.{i}.layers.1.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_bbox_head.{i}.layers.1.bias",
                f"model.decoder.bbox_embed.{i}.layers.1.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_bbox_head.{i}.layers.2.weight",
                f"model.decoder.bbox_embed.{i}.layers.2.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.dec_bbox_head.{i}.layers.2.bias",
                f"model.decoder.bbox_embed.{i}.layers.2.bias",
            )
        )

    # decoder projection
    for i in range(len(config.decoder_in_channels)):
        rename_keys.append(
            (
                f"decoder.input_proj.{i}.conv.weight",
                f"model.decoder_input_proj.{i}.0.weight",
            )
        )
        for last in last_key:
            rename_keys.append(
                (
                    f"decoder.input_proj.{i}.norm.{last}",
                    f"model.decoder_input_proj.{i}.1.{last}",
                )
            )

    # convolutional projection + query embeddings + layernorm of decoder + class and bounding box heads
    rename_keys.extend(
        [
            ("decoder.denoising_class_embed.weight", "model.denoising_class_embed.weight"),
            ("decoder.query_pos_head.layers.0.weight", "model.decoder.query_pos_head.layers.0.weight"),
            ("decoder.query_pos_head.layers.0.bias", "model.decoder.query_pos_head.layers.0.bias"),
            ("decoder.query_pos_head.layers.1.weight", "model.decoder.query_pos_head.layers.1.weight"),
            ("decoder.query_pos_head.layers.1.bias", "model.decoder.query_pos_head.layers.1.bias"),
            ("decoder.enc_output.proj.weight", "model.enc_output.0.weight"),
            ("decoder.enc_output.proj.bias", "model.enc_output.0.bias"),
            ("decoder.enc_output.norm.weight", "model.enc_output.1.weight"),
            ("decoder.enc_output.norm.bias", "model.enc_output.1.bias"),
            ("decoder.enc_score_head.weight", "model.enc_score_head.weight"),
            ("decoder.enc_score_head.bias", "model.enc_score_head.bias"),
            ("decoder.enc_bbox_head.layers.0.weight", "model.enc_bbox_head.layers.0.weight"),
            ("decoder.enc_bbox_head.layers.0.bias", "model.enc_bbox_head.layers.0.bias"),
            ("decoder.enc_bbox_head.layers.1.weight", "model.enc_bbox_head.layers.1.weight"),
            ("decoder.enc_bbox_head.layers.1.bias", "model.enc_bbox_head.layers.1.bias"),
            ("decoder.enc_bbox_head.layers.2.weight", "model.enc_bbox_head.layers.2.weight"),
            ("decoder.enc_bbox_head.layers.2.bias", "model.enc_bbox_head.layers.2.bias"),
        ]
    )

    return rename_keys


def rename_key(state_dict, old, new):
    try:
        val = state_dict.pop(old)
        state_dict[new] = val
    except Exception:
        pass


def read_in_q_k_v(state_dict, config):
    prefix = ""
    encoder_hidden_dim = config.encoder_hidden_dim

    # first: transformer encoder
    for i in range(config.encoder_layers):
        # read in weights + bias of input projection layer (in PyTorch's MultiHeadAttention, this is a single matrix + bias)
        in_proj_weight = state_dict.pop(f"{prefix}encoder.encoder.{i}.layers.0.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}encoder.encoder.{i}.layers.0.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.q_proj.weight"] = in_proj_weight[
            :encoder_hidden_dim, :
        ]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.q_proj.bias"] = in_proj_bias[:encoder_hidden_dim]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.k_proj.weight"] = in_proj_weight[
            encoder_hidden_dim : 2 * encoder_hidden_dim, :
        ]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.k_proj.bias"] = in_proj_bias[
            encoder_hidden_dim : 2 * encoder_hidden_dim
        ]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.v_proj.weight"] = in_proj_weight[
            -encoder_hidden_dim:, :
        ]
        state_dict[f"model.encoder.encoder.{i}.layers.0.self_attn.v_proj.bias"] = in_proj_bias[-encoder_hidden_dim:]
    # next: transformer decoder (which is a bit more complex because it also includes cross-attention)
    for i in range(config.decoder_layers):
        # read in weights + bias of input projection layer of self-attention
        in_proj_weight = state_dict.pop(f"{prefix}decoder.decoder.layers.{i}.self_attn.in_proj_weight")
        in_proj_bias = state_dict.pop(f"{prefix}decoder.decoder.layers.{i}.self_attn.in_proj_bias")
        # next, add query, keys and values (in that order) to the state dict
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.weight"] = in_proj_weight[:256, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.q_proj.bias"] = in_proj_bias[:256]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.weight"] = in_proj_weight[256:512, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.k_proj.bias"] = in_proj_bias[256:512]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.weight"] = in_proj_weight[-256:, :]
        state_dict[f"model.decoder.layers.{i}.self_attn.v_proj.bias"] = in_proj_bias[-256:]


# We will verify our results on an image of cute cats
def prepare_img():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    im = Image.open(requests.get(url, stream=True).raw)

    return im


@torch.no_grad()
def convert_rt_detr_v2_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub, repo_id):
    """
    Copy/paste/tweak model's weights to our RTDETR structure.
    """

    # load default config
    config = get_rt_detr_v2_config(model_name)

    # load original model from torch hub
    model_name_to_checkpoint_url = {
        "rtdetr_v2_r18vd": "https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth",
        "rtdetr_v2_r34vd": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r34vd_120e_coco_ema.pth",
        "rtdetr_v2_r50vd_m": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_m_7x_coco_ema.pth",
        "rtdetr_v2_r50vd": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth",
        "rtdetr_v2_r101vd": "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r101vd_6x_coco_from_paddle.pth",
    }
    logger.info(f"Converting model {model_name}...")
    state_dict = torch.hub.load_state_dict_from_url(model_name_to_checkpoint_url[model_name], map_location="cpu")[
        "ema"
    ]["module"]

    # rename keys
    for src, dest in create_rename_keys(config):
        rename_key(state_dict, src, dest)
    # query, key and value matrices need special treatment
    read_in_q_k_v(state_dict, config)
    # important: we need to prepend a prefix to each of the base model keys as the head models use different attributes for them
    for key in state_dict.copy().keys():
        if key.endswith("num_batches_tracked"):
            del state_dict[key]
        # for two_stage
        if "bbox_embed" in key or ("class_embed" in key and "denoising_" not in key):
            state_dict[key.split("model.decoder.")[-1]] = state_dict[key]

    # no need in ckpt
    del state_dict["decoder.anchors"]
    del state_dict["decoder.valid_mask"]

    # finally, create HuggingFace model and load state dict
    model = RtDetrV2ForObjectDetection(config)
    model.load_state_dict(state_dict)
    model.eval()

    # load image processor
    image_processor = RTDetrImageProcessor()

    # prepare image
    img = prepare_img()

    # preprocess image
    transformations = transforms.Compose(
        [
            transforms.Resize([640, 640], interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    original_pixel_values = transformations(img).unsqueeze(0)  # insert batch dimension

    encoding = image_processor(images=img, return_tensors="pt")
    pixel_values = encoding["pixel_values"]

    assert torch.allclose(original_pixel_values, pixel_values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pixel_values = pixel_values.to(device)

    # Pass image by the model
    outputs = model(pixel_values)

    if model_name == "rtdetr_v2_r18vd":
        expected_slice_logits = torch.tensor(
            [[-3.7045, -5.1913, -6.1787], [-4.0106, -9.3450, -5.2043], [-4.1287, -4.7463, -5.8634]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.2582, 0.5497, 0.4764], [0.1684, 0.1985, 0.2120], [0.7665, 0.4146, 0.4669]]
        )
    elif model_name == "rtdetr_v2_r34vd":
        expected_slice_logits = torch.tensor(
            [[-4.6108, -5.9453, -3.8505], [-3.8702, -6.1136, -5.5677], [-3.7790, -6.4538, -5.9449]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.1691, 0.1984, 0.2118], [0.2594, 0.5506, 0.4736], [0.7669, 0.4136, 0.4654]]
        )
    elif model_name == "rtdetr_v2_r50vd_m":
        expected_slice_logits = torch.tensor(
            [[-2.7453, -5.4595, -7.3702], [-3.1858, -5.3803, -7.9838], [-5.0293, -7.0083, -4.2888]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.7711, 0.4135, 0.4577], [0.2570, 0.5480, 0.4755], [0.1694, 0.1992, 0.2127]]
        )
    elif model_name == "rtdetr_v2_r50vd":
        expected_slice_logits = torch.tensor(
            [[-4.7881, -4.6754, -6.1624], [-5.4441, -6.6486, -4.3840], [-3.5455, -4.9318, -6.3544]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.2588, 0.5487, 0.4747], [0.5497, 0.2760, 0.0573], [0.7688, 0.4133, 0.4634]]
        )
    elif model_name == "rtdetr_v2_r101vd":
        expected_slice_logits = torch.tensor(
            [[-4.6162, -4.9189, -4.6656], [-4.4701, -4.4997, -4.9659], [-5.6641, -7.9000, -5.0725]]
        )
        expected_slice_boxes = torch.tensor(
            [[0.7707, 0.4124, 0.4585], [0.2589, 0.5492, 0.4735], [0.1688, 0.1993, 0.2108]]
        )
    else:
        raise ValueError(f"Unknown rt_detr_v2_name: {model_name}")

    assert torch.allclose(outputs.logits[0, :3, :3], expected_slice_logits.to(outputs.logits.device), atol=1e-4)
    assert torch.allclose(outputs.pred_boxes[0, :3, :3], expected_slice_boxes.to(outputs.pred_boxes.device), atol=1e-3)

    if pytorch_dump_folder_path is not None:
        Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
        print(f"Saving model {model_name} to {pytorch_dump_folder_path}")
        model.save_pretrained(pytorch_dump_folder_path)
        print(f"Saving image processor to {pytorch_dump_folder_path}")
        image_processor.save_pretrained(pytorch_dump_folder_path)

    if push_to_hub:
        # Upload model, image processor and config to the hub
        logger.info("Uploading PyTorch model and image processor to the hub...")
        config.push_to_hub(
            repo_id=repo_id,
            commit_message="Add config from convert_rt_detr_v2_original_pytorch_checkpoint_to_pytorch.py",
        )
        model.push_to_hub(
            repo_id=repo_id,
            commit_message="Add model from convert_rt_detr_v2_original_pytorch_checkpoint_to_pytorch.py",
        )
        image_processor.push_to_hub(
            repo_id=repo_id,
            commit_message="Add image processor from convert_rt_detr_v2_original_pytorch_checkpoint_to_pytorch.py",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="rtdetr_v2_r18vd",
        type=str,
        help="model_name of the checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to the hub or not.")
    parser.add_argument(
        "--repo_id",
        type=str,
        help="repo_id where the model will be pushed to.",
    )
    args = parser.parse_args()
    convert_rt_detr_v2_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub, args.repo_id)
