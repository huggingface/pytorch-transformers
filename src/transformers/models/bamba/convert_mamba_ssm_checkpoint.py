# coding=utf-8
# Copyright 2024 IBM and the HuggingFace Inc. team. All rights reserved.
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
"""This script can be used to convert checkpoints provided in the `mamba_ssm` library into the format provided in HuggingFace `transformers`. It depends on the `mamba2_ssm` package to be installed."""

import argparse
import json
import os
import re
from os import path
from typing import Dict, Union

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file

from transformers import AutoTokenizer
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME

from .configuration_bamba import BambaConfig


def convert_state_dict_from_mamba_ssm(original_sd: Dict) -> Dict[str, torch.Tensor]:
    state_dict = {}

    for orig_k, param in original_sd.items():
        k = orig_k.replace("backbone", "model")

        # for embeddings
        k = k.replace("embedding", "embed_tokens")

        # for mixer
        k = k.replace("mixer", "mamba")

        # for final layernorm
        k = k.replace("norm_f", "final_layernorm")

        # for block layernorm
        k = re.sub(r"(\d+)\.norm\.", r"\1.input_layernorm.", k)
        k = re.sub(r"(\d+)\.norm2\.", r"\1.pre_ff_layernorm.", k)

        # for mlp
        k = k.replace("mlp.fc2", "feed_forward.down_proj")

        if "mlp.fc1" in k:
            param, param2 = torch.chunk(param, 2, dim=0)
            k2 = k.replace("mlp.fc1", "feed_forward.gate_proj")
            state_dict[k2] = param2
            k = k.replace("mlp.fc1", "feed_forward.up_proj")

        if ("in_proj" in k and orig_k.replace("in_proj", "conv1d") in original_sd) or (
            "out_proj" in k and orig_k.replace("out_proj", "conv1d") in original_sd
        ):
            # then this must be a mamba
            pass
        else:
            # for attn
            # - because mixer was replaced to mamba above
            k = k.replace("mamba.out_proj", "self_attn.o_proj")
            if "mamba.in_proj" in k:
                m, n = param.shape
                d = (m - n) // 2
                param, param2, param3 = torch.split(param, [n, d, d], dim=0)
                k2 = k.replace("mamba.in_proj", "self_attn.k_proj")
                state_dict[k2] = param2
                k2 = k.replace("mamba.in_proj", "self_attn.v_proj")
                state_dict[k2] = param3
                k = k.replace("mamba.in_proj", "self_attn.q_proj")

        state_dict[k] = param

    return state_dict


# Adapted from transformers.models.mamba.convert_mamba_ssm_checkpoint_to_pytorch.py
def convert_ssm_config_to_hf_config(
    config_ssm: Dict,
    **kwargs,
) -> BambaConfig:
    """Convert a config from mamba_ssm to a BambaConfig from here."""
    hf_config: BambaConfig = BambaConfig(**kwargs)

    hf_config.architectures = ["BambaForCausalLM"]

    # Set important values from config and recalculate other resulting entries
    hf_config.hidden_size = config_ssm["d_model"]
    hf_config.intermediate_size = config_ssm["d_intermediate"]
    hf_config.mamba_n_heads = (hf_config.hidden_size * hf_config.mamba_expand) // hf_config.mamba_d_head
    hf_config.num_hidden_layers = config_ssm["n_layer"]
    hf_config.tie_word_embeddings = config_ssm["tie_embeddings"]

    # currently this script assumes config_ssm belongs to v2
    if config_ssm["ssm_cfg"].get("layer") != "Mamba2":
        raise ValueError("Conversion script only supports Mamba2")

    # Set attention values
    attn_cfg = config_ssm.get("attn_cfg")
    if attn_cfg:
        assert attn_cfg["causal"], "Only support non-causal attention."
        assert not attn_cfg["qkv_proj_bias"], "Only support no qkv bias."
        assert not attn_cfg["out_proj_bias"], "Only support no out bias."
        hf_config.attn_rotary_emb = attn_cfg["rotary_emb_dim"]
        hf_config.num_attention_heads = attn_cfg["num_heads"]
        hf_config.num_key_value_heads = attn_cfg["num_heads_kv"]

    attention_layer_indices = config_ssm.get("attn_layer_idx")
    if attention_layer_indices:
        hf_config.attn_layer_indices = attention_layer_indices

    # Padded vocab size, mostly of 16 but 32 is also very common in different models
    vocab_size = config_ssm["vocab_size"]
    pad_vocab_size_multiple = config_ssm["pad_vocab_size_multiple"]
    if (vocab_size % pad_vocab_size_multiple) != 0:
        vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
    hf_config.vocab_size = vocab_size

    return hf_config


def save_single_safetensor(
    state_dict: Dict,
    save_directory: str,
    metadata: Dict,
):
    save_file(
        state_dict,
        os.path.join(save_directory, SAFE_WEIGHTS_NAME),
        metadata,
    )


def save_sharded_safetensors(
    state_dict: Dict,
    save_directory: str,
    metadata: Dict,
    max_shard_size: Union[int, str] = "5GB",
):
    filename_pattern = SAFE_WEIGHTS_NAME.replace(".bin", "{suffix}.bin").replace(
        ".safetensors", "{suffix}.safetensors"
    )
    state_dict_split = split_torch_state_dict_into_shards(
        state_dict, filename_pattern=filename_pattern, max_shard_size=max_shard_size
    )
    index = {
        "metadata": state_dict_split.metadata,
        "weight_map": state_dict_split.tensor_to_filename,
    }
    # Save the index
    with open(os.path.join(save_directory, SAFE_WEIGHTS_INDEX_NAME), "w", encoding="utf-8") as f:
        content = json.dumps(index, indent=2, sort_keys=True) + "\n"
        f.write(content)

    filename_to_tensors = state_dict_split.filename_to_tensors.items()
    for shard_file, tensors in filename_to_tensors:
        shard = {tensor: state_dict[tensor].contiguous() for tensor in tensors}
        save_file(shard, os.path.join(save_directory, shard_file), metadata=metadata)


# Adapted from transformers.models.mamba.convert_mamba_ssm_checkpoint_to_pytorch.py
def convert_mamba_ssm_checkpoint_file_to_huggingface_model_file(
    mamba_ssm_checkpoint_path: str,
    precision: str,
    output_dir: str,
    tokenizer_path: str = None,
    save_model: Union[bool, str] = True,
) -> None:
    # load tokenizer if provided, this will be used to set the
    # token_ids in the config file
    token_ids = {}
    if tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        for key in [
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
        ]:
            id = getattr(tokenizer, key, None)
            if id:
                token_ids[key] = id

    # there are some configs unsettable by mamba_ssn config, so
    # if there are changes from the defaults, have to pass them into
    # the function
    unsettables = {
        "mamba_d_head": 64,
        "mamba_d_state": 128,
        "mamba_n_groups": 1,
        "rms_norm_eps": 1e-5,
    }

    # Load and save config based on name
    config_path = path.join(mamba_ssm_checkpoint_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as json_file:
        config = json.load(json_file)

    # convert the config
    hf_config = convert_ssm_config_to_hf_config(
        config_ssm=config,
        **token_ids,
        **unsettables,
    )
    hf_config.save_pretrained(output_dir)

    # Load state dict of the original model and transfer to hf model
    state_dict = torch.load(
        path.join(mamba_ssm_checkpoint_path, "pytorch_model.bin"),
        map_location="cpu",
        weights_only=True,
    )
    # FIXME: allow other parameters to pass in
    state_dict = convert_state_dict_from_mamba_ssm(state_dict)

    # Save new model to pytorch_dump_path
    dtype = torch.float32 if precision == "fp32" else (torch.bfloat16 if precision == "bf16" else torch.float16)

    save_file_fn = None
    if isinstance(save_model, bool) and save_model:
        save_file_fn = save_single_safetensor
    elif isinstance(save_model, str) and save_model == "sharded":
        save_file_fn = save_sharded_safetensors

    if save_file_fn:
        save_file_fn({k: v.to(dtype) for k, v in state_dict.items()}, output_dir, metadata={"format": "pt"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--mamba_ssm_checkpoint_directory",
        type=str,
        required=True,
        help="Path to a directory containing the `pytorch_model.bin` mamba_ssm checkpoint file to be converted.",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        default="fp16",
        const="fp16",
        required=True,
        choices=("fp32", "fp16", "bf16"),
        help="The precision the model will be saved in. Select from fp32, fp16 or bf16.",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, required=True, help="Path to directory to save the converted output model to."
    )
    parser.add_argument(
        "-t",
        "--tokenizer_model_path",
        type=str,
        default=None,
        required=False,
        help="Path to a the tokenizer file.",
    )
    args = parser.parse_args()

    convert_mamba_ssm_checkpoint_file_to_huggingface_model_file(
        args.mamba2_checkpoint_directory,
        args.precision,
        args.output_dir,
    )
