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
"""Convert D-FINE checkpoints with Timm backbone"""

import argparse
import json
from pathlib import Path
import glob

import requests
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from torchvision import transforms

from transformers import DFineConfig, DFineForObjectDetection, RTDetrImageProcessor
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def get_d_fine_config(model_name: str) -> DFineConfig:
    config = DFineConfig()

    config.num_labels = 80
    repo_id = "huggingface/label-files"
    filename = "coco-detection-mmdet-id2label.json"
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    config.id2label = id2label
    config.label2id = {v: k for k, v in id2label.items()}

    config.backbone_config.hidden_sizes = [64, 128, 256, 512]
    config.backbone_config.depths = [5, 5, 5, 5]
    config.backbone_config.layer_type = "basic"
    config.backbone_config.embedding_size = 32
    config.encoder_in_channels = [512, 1024, 2048]
    config.hidden_expansion = 1.0
    config.decoder_layers = 6


    return config


def load_original_state_dict(repo_id):
    directory_path = snapshot_download(repo_id=repo_id, allow_patterns=["*.pth"])

    original_state_dict = {}
    for path in glob.glob(f"{directory_path}/*"):
        if path.endswith(".pth"):
            model = torch.load(path, map_location="cpu")['model']
            for key in model.keys():
                original_state_dict[key] = model[key]
            break

    return original_state_dict


def create_rename_keys(config):
    # here we list all keys to be renamed (original name on the left, our name on the right)
    rename_keys = []

    # stem
    # fmt: off
    last_key = ["weight", "bias", "running_mean", "running_var"]

    #done
    for level in range(4):
        if level + 1 == 2:
            rename_keys.append((f"backbone.stem.stem{level+1}a.conv.weight", f"model.backbone.model.embedder.embedder.{level}.convolution.weight"))
            rename_keys.append((f"backbone.stem.stem{level+1}b.conv.weight", f"model.backbone.model.embedder.embedder.{level+1}.convolution.weight"))
        else:
            rename_keys.append((f"backbone.stem.stem{level+1}.conv.weight", f"model.backbone.model.embedder.embedder.{level+1}.convolution.weight"))
        for last in last_key:
            if level + 1 == 2:
                rename_keys.append((f"backbone.stem.stem{level+1}a.bn.{last}", f"model.backbone.model.embedder.embedder.{level}.normalization.{last}"))
                rename_keys.append((f"backbone.stem.stem{level+1}b.bn.{last}", f"model.backbone.model.embedder.embedder.{level+1}.normalization.{last}"))
            else:
                rename_keys.append((f"backbone.stem.stem{level+1}.bn.{last}", f"model.backbone.model.embedder.embedder.{level+1}.normalization.{last}"))

    for stage_idx in range(len(config.backbone_config.depths)):
        for layer_idx in range(config.backbone_config.depths[stage_idx]):
            # shortcut
            if stage_idx == 0 or stage_idx == 3:
                rename_keys.append(
                    (
                        f"backbone.stages.{stage_idx}.blocks.0.layers.{layer_idx}.conv.weight",
                        f"model.backbone.model.encoder.stages.{stage_idx}.blocks.0.layers.{layer_idx}.shortcut.convolution.weight",
                    )
                )
                for last in last_key:
                    rename_keys.append(
                    (
                        f"backbone.stages.{stage_idx}.blocks.0.layers.{layer_idx}.bn.{last}",
                        f"model.backbone.model.encoder.stages.{stage_idx}.blocks.0.layers.{layer_idx}.shortcut.normalization.{last}",
                    )
                )
                #layers
                rename_keys.append(
                    (
                        f"backbone.stages.{stage_idx}.blocks.0.layers.{layer_idx}.conv1.conv.weigh",
                        f"model.backbone.model.encoder.stages.{stage_idx}.blocks.0.layers.{layer_idx}.layer.0.convolution.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.stages.{stage_idx}.blocks.0.layers.{layer_idx}.conv2.conv.weigh",
                        f"model.backbone.model.encoder.stages.{stage_idx}.blocks.0.layers.{layer_idx}.layer.1.convolution.weight",
                    )
                )
                for last in last_key:
                    rename_keys.append((
                        f"backbone.stages.{stage_idx}.blocks.0.layers.{layer_idx}.conv1.bn.{last}",
                        f"model.backbone.model.encoder.stages.{stage_idx}.blocks.0.layers.{layer_idx}.layer.0.normalization.{last}",
                        ))

                for last in last_key:
                    rename_keys.append((
                        f"backbone.stages.{stage_idx}.blocks.0.{layer_idx}.conv2.bn.{last}",
                        f"model.backbone.model.encoder.stages.{stage_idx}.blocks.0.layers.{layer_idx}.layer.1.normalization.{last}",
                        ))
    
            for b in range(1, 4):
                rename_keys.append(
                    (
                        f"backbone.stages.{stage_idx}.blocks.{b}.layers.{layer_idx}.conv.weight",
                        f"model.backbone.model.encoder.stages.{stage_idx}.blocks.{b}.layers.{layer_idx}.shortcut.convolution.weight",
                    )
                )
                for last in last_key:
                    rename_keys.append(
                    (
                        f"backbone.stages.{stage_idx}.blocks.{b}.layers.{layer_idx}.bn.{last}",
                        f"model.backbone.model.encoder.stages.{stage_idx}.blocks.{b}.layers.{layer_idx}.shortcut.normalization.{last}",
                    )
                )
                #layers
                rename_keys.append(
                    (
                        f"backbone.stages.{stage_idx}.blocks.{b}.layers.{layer_idx}.conv1.conv.weigh",
                        f"model.backbone.model.encoder.stages.{stage_idx}.blocks.{b}.layers.{layer_idx}.layer.0.convolution.weight",
                    )
                )
                rename_keys.append(
                    (
                        f"backbone.stages.{stage_idx}.blocks.{b}.layers.{layer_idx}.conv2.conv.weigh",
                        f"model.backbone.model.encoder.stages.{stage_idx}.blocks.{b}.layers.{layer_idx}.layer.1.convolution.weight",
                    )
                )
                for last in last_key:
                    rename_keys.append((
                        f"backbone.stages.{stage_idx}.blocks.{b}.layers.{layer_idx}.conv1.bn.{last}",
                        f"model.backbone.model.encoder.stages.{stage_idx}.blocks.{b}.layers.{layer_idx}.layer.0.normalization.{last}",
                        ))

                for last in last_key:
                    rename_keys.append((
                        f"backbone.stages.{stage_idx}.blocks.{b}.{layer_idx}.conv2.bn.{last}",
                        f"model.backbone.model.encoder.stages.{stage_idx}.blocks.{b}.layers.{layer_idx}.layer.1.normalization.{last}",
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

    for i in range(len(config.encoder_in_channels) - 1):
        # encoder layers: hybridencoder parts
        # number of cv is 4 according to original implementation
        for j in range(5):
            rename_keys.append(
                (f"encoder.fpn_blocks.{i}.cv{j}.conv.weight", f"model.encoder.fpn_blocks.{i}.cv{j}.conv.weight")
            )
            for last in last_key:
                rename_keys.append(
                    (
                        f"encoder.fpn_blocks.{i}.cv{j}.norm.{last}",
                        f"model.encoder.fpn_blocks.{i}.cv{j}.norm.{last}",
                    )
                )

        rename_keys.append((f"encoder.lateral_convs.{i}.conv.weight", f"model.encoder.lateral_convs.{i}.conv.weight"))
        for last in last_key:
            rename_keys.append(
                (f"encoder.lateral_convs.{i}.norm.{last}", f"model.encoder.lateral_convs.{i}.norm.{last}")
            )

        for c in range(2, 4):
            for j in range(3):
                for k in range(1, 3):
                    rename_keys.append(
                        (
                            f"encoder.fpn_blocks.{i}.cv{c}.0.bottlenecks.{j}.conv{k}.conv.weight",
                            f"model.encoder.fpn_blocks.{i}.cv{c}.0.bottlenecks.{j}.conv{k}.conv.weight",
                        )
                    )
                    for last in last_key:
                        rename_keys.append(
                            (
                                f"encoder.fpn_blocks.{i}.cv{c}.0.bottlenecks.{j}.conv{k}.norm.{last}",
                                f"model.encoder.fpn_blocks.{i}.cv{c}.0.bottlenecks.{j}.conv{k}.norm.{last}",
                            )
                        )
                    rename_keys.append(
                        (
                            f"encoder.fpn_blocks.{i}.cv{c}.1.bottlenecks.{j}.conv{k}.conv.weight",
                            f"model.encoder.fpn_blocks.{i}.cv{c}.1.bottlenecks.{j}.conv{k}.conv.weight",
                        )
                    )
                    for last in last_key:
                        rename_keys.append(
                            (
                                f"encoder.fpn_blocks.{i}.cv{c}.1.bottlenecks.{j}.conv{k}.norm.{last}",
                                f"model.encoder.fpn_blocks.{i}.cv{c}.1.bottlenecks.{j}.conv{k}.norm.{last}",
                            )
                        )

        for j in range(5):
            rename_keys.append(
                (f"encoder.pan_blocks.{i}.cv{j}.conv.weight", f"model.encoder.pan_blocks.{i}.cv{j}.conv.weight")
            )
            for last in last_key:
                rename_keys.append(
                    (
                        f"encoder.pan_blocks.{i}.cv{j}.norm.{last}",
                        f"model.encoder.pan_blocks.{i}.cv{j}.norm.{last}",
                    )
                )

        for c in range(2, 4):
            for j in range(3):
                for k in range(1, 3):
                    rename_keys.append(
                        (
                            f"encoder.pan_blocks.{i}.cv{c}.0.bottlenecks.{j}.conv{k}.conv.weight",
                            f"model.encoder.pan_blocks.{i}.cv{c}.0.bottlenecks.{j}.conv{k}.conv.weight",
                        )
                    )
                    for last in last_key:
                        rename_keys.append(
                            (
                                f"encoder.pan_blocks.{i}.cv{c}.0.bottlenecks.{j}.conv{k}.norm.{last}",
                                f"model.encoder.pan_blocks.{i}.cv{c}.0.bottlenecks.{j}.conv{k}.norm.{last}",
                            )
                        )

        rename_keys.append(
            (f"encoder.downsample_convs.{i}.0.cv{i}.conv.weight", f"model.encoder.downsample_convs.{i}.conv.weight")
        )
        for last in last_key:
            rename_keys.append(
                (f"encoder.downsample_convs.{i}.0.cv{i}.norm.{last}", f"model.encoder.downsample_convs.{i}.norm.{last}")
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
                f"decoder.decoder.layers.{i}.cross_attn.num_points_scale",
                f"model.decoder.layers.{i}.encoder_attn.num_points_scale",
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
                f"decoder.decoder.layers.{i}.gateway.gate.weight",
                f"model.decoder.layers.{i}.gateway.gate.weights",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.gateway.gate.bias",
                f"model.decoder.layers.{i}.gateway.gate.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.gateway.norm.weight",
                f"model.decoder.layers.{i}.gateway.norm.weights",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.layers.{i}.gateway.norm.bias",
                f"model.decoder.layers.{i}.gateway.norm.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.lqe_layers.{i}.reg_conf.layers.0.weight",
                f"model.decoder.lqe_layers.{i}.reg_conf.layers.0.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.lqe_layers.{i}.reg_conf.layers.0.bias",
                f"model.decoder.lqe_layers.{i}.reg_conf.layers.0.bias",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.lqe_layers.{i}.reg_conf.layers.1.weight",
                f"model.decoder.lqe_layers.{i}.reg_conf.layers.1.weight",
            )
        )
        rename_keys.append(
            (
                f"decoder.decoder.lqe_layers.{i}.reg_conf.layers.1.bias",
                f"model.decoder.lqe_layers.{i}.reg_conf.layers.1.bias",
            )
        )
       
        rename_keys.append(
            (f"decoder.decoder.layers.{i}.norm1.weight", f"model.decoder.layers.{i}.self_attn_layer_norm.weight")
        )
        rename_keys.append(
            (f"decoder.decoder.layers.{i}.norm1.bias", f"model.decoder.layers.{i}.self_attn_layer_norm.bias")
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

    # convolutional projection + query embeddings + layernorm of decoder + class and bounding box heads
    rename_keys.extend(
        [
            ("decoder.denoising_class_embed.weight", "model.denoising_class_embed.weight"),
            ("decoder.query_pos_head.layers.0.weight", "model.decoder.query_pos_head.layers.0.weight"),
            ("decoder.query_pos_head.layers.0.bias", "model.decoder.query_pos_head.layers.0.bias"),
            ("decoder.query_pos_head.layers.1.weight", "model.decoder.query_pos_head.layers.1.weight"),
            ("decoder.query_pos_head.layers.1.bias", "model.decoder.query_pos_head.layers.1.bias"),
            ("decoder.enc_output.0.weight", "model.enc_output.0.weight"),
            ("decoder.enc_output.0.bias", "model.enc_output.0.bias"),
            ("decoder.enc_output.1.weight", "model.enc_output.1.weight"),
            ("decoder.enc_output.1.bias", "model.enc_output.1.bias"),
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
def convert_rt_detr_checkpoint(model_name, pytorch_dump_folder_path, push_to_hub, repo_id):
    """
    Copy/paste/tweak model's weights to our RTDETR structure.
    """

    # load default config
    config = get_d_fine_config(model_name)
    state_dict = load_original_state_dict(repo_id)
    model = DFineForObjectDetection(config)
    logger.info(f"Converting model {model_name}...")

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

    # finally, create HuggingFace model and load state dict
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

    if model_name == "rtdetr_r18vd":
        expected_slice_logits = torch.tensor(
            [
                [-4.3364253, -6.465683, -3.6130402],
                [-4.083815, -6.4039373, -6.97881],
                [-4.192215, -7.3410473, -6.9027247],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.16868353, 0.19833282, 0.21182671],
                [0.25559652, 0.55121744, 0.47988364],
                [0.7698693, 0.4124569, 0.46036878],
            ]
        )
    elif model_name == "rtdetr_r34vd":
        expected_slice_logits = torch.tensor(
            [
                [-4.3727384, -4.7921476, -5.7299604],
                [-4.840536, -8.455345, -4.1745796],
                [-4.1277084, -5.2154565, -5.7852697],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.258278, 0.5497808, 0.4732004],
                [0.16889669, 0.19890057, 0.21138911],
                [0.76632994, 0.4147879, 0.46851268],
            ]
        )
    elif model_name == "rtdetr_r50vd_m":
        expected_slice_logits = torch.tensor(
            [
                [-4.319764, -6.1349025, -6.094794],
                [-5.1056995, -7.744766, -4.803956],
                [-4.7685347, -7.9278393, -4.5751696],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.2582739, 0.55071366, 0.47660282],
                [0.16811174, 0.19954777, 0.21292639],
                [0.54986024, 0.2752091, 0.0561416],
            ]
        )
    elif model_name == "rtdetr_r50vd":
        expected_slice_logits = torch.tensor(
            [
                [-4.6476398, -5.001154, -4.9785104],
                [-4.1593494, -4.7038546, -5.946485],
                [-4.4374595, -4.658361, -6.2352347],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.16880608, 0.19992264, 0.21225442],
                [0.76837635, 0.4122631, 0.46368608],
                [0.2595386, 0.5483334, 0.4777486],
            ]
        )
    elif model_name == "rtdetr_r101vd":
        expected_slice_logits = torch.tensor(
            [
                [-4.6162, -4.9189, -4.6656],
                [-4.4701, -4.4997, -4.9659],
                [-5.6641, -7.9000, -5.0725],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.7707, 0.4124, 0.4585],
                [0.2589, 0.5492, 0.4735],
                [0.1688, 0.1993, 0.2108],
            ]
        )
    elif model_name == "rtdetr_r18vd_coco_o365":
        expected_slice_logits = torch.tensor(
            [
                [-4.8726, -5.9066, -5.2450],
                [-4.8157, -6.8764, -5.1656],
                [-4.7492, -5.7006, -5.1333],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.2552, 0.5501, 0.4773],
                [0.1685, 0.1986, 0.2104],
                [0.7692, 0.4141, 0.4620],
            ]
        )
    elif model_name == "rtdetr_r50vd_coco_o365":
        expected_slice_logits = torch.tensor(
            [
                [-4.6491, -3.9252, -5.3163],
                [-4.1386, -5.0348, -3.9016],
                [-4.4778, -4.5423, -5.7356],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.2583, 0.5492, 0.4747],
                [0.5501, 0.2754, 0.0574],
                [0.7693, 0.4137, 0.4613],
            ]
        )
    elif model_name == "rtdetr_r101vd_coco_o365":
        expected_slice_logits = torch.tensor(
            [
                [-4.5152, -5.6811, -5.7311],
                [-4.5358, -7.2422, -5.0941],
                [-4.6919, -5.5834, -6.0145],
            ]
        )
        expected_slice_boxes = torch.tensor(
            [
                [0.7703, 0.4140, 0.4583],
                [0.1686, 0.1991, 0.2107],
                [0.2570, 0.5496, 0.4750],
            ]
        )
    else:
        raise ValueError(f"Unknown rt_detr_name: {model_name}")

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
            repo_id=repo_id, commit_message="Add config from convert_rt_detr_original_pytorch_checkpoint_to_pytorch.py"
        )
        model.push_to_hub(
            repo_id=repo_id, commit_message="Add model from convert_rt_detr_original_pytorch_checkpoint_to_pytorch.py"
        )
        image_processor.push_to_hub(
            repo_id=repo_id,
            commit_message="Add image processor from convert_rt_detr_original_pytorch_checkpoint_to_pytorch.py",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="D-FINE",
        type=str,
        help="model_name of the checkpoint you'd like to convert.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model directory."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether to push the model to the hub or not.")
    parser.add_argument(
        "--repo_id",
        default="Peterande/D-FINE",
        type=str,
        help="repo_id where the model will be pushed to.",
    )
    args = parser.parse_args()
    convert_rt_detr_checkpoint(args.model_name, args.pytorch_dump_folder_path, args.push_to_hub, args.repo_id)
