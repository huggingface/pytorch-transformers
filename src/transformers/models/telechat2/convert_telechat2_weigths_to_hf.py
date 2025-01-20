import argparse
import json
import os
import re

import torch
from safetensors.torch import load_file

from transformers import TeleChat2Config, TeleChat2ForCausalLM


# fmt: off
# `None` means we drop the key
STATE_DICT_MAPPING = {
    # Model keys
    r"transformer.word_embeddings.weight":                               r"model.embed_tokens.weight",
    r"transformer.ln_f.weight":                                          r"model.norm.weight",

    # Layers keys
    r"transformer.h.(\d+).input_layernorm.weight":                       r"model.layers.\1.input_layernorm.weight",
    r"transformer.h.(\d+).post_attention_layernorm.weight":              r"model.layers.\1.post_attention_layernorm.weight",

    # Attention keys
    r"transformer.h.(\d+).self_attention.dense.weight":                  r"model.layers.\1.self_attn.o_proj.weight",
    # qkv_proj will later be split in q|k|v|_proj
    r"transformer.h.(\d+).self_attention.key_value.(weight|bias)":       r"model.layers.\1.self_attn.key_value.\2",
    r"transformer.h.(\d+).self_attention.query.(weight|bias)":           r"model.layers.\1.self_attn.query.\2",

    # MLP keys
    r"transformer.h.(\d+).mlp.gate_proj.weight":                     r"model.layers.\1.mlp.gate_proj.weight",
    r"transformer.h.(\d+).mlp.up_proj.weight":                     r"model.layers.\1.mlp.up_proj.weight",
    r"transformer.h.(\d+).mlp.down_proj.weight":                     r"model.layers.\1.mlp.down_proj.weight",
}
# fmt: on


def load_weights(input_dir: str):
    safetensor_files = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if x.endswith(".safetensors")]
    bin_files = [os.path.join(input_dir, x) for x in os.listdir(input_dir) if x.endswith(".bin")]

    all_weights = {}

    if safetensor_files:
        safetensor_files = sorted(safetensor_files, key=lambda x: int(x.rsplit("-", 3)[1]))
        for file in safetensor_files:
            tensors = load_file(file)
            all_weights.update(tensors)
        return all_weights

    elif bin_files:
        bin_files = sorted(bin_files, key=lambda x: int(x.rsplit("-", 3)[1]))
        for file in bin_files:
            tensors = torch.load(file, map_location="cpu")
            all_weights.update(tensors)
        return all_weights

    else:
        raise ValueError("No .safetensors or .bin files found in the specified directory.")


def map_old_key_to_new(old_key):
    for pattern, replacement in STATE_DICT_MAPPING.items():
        if replacement is None:
            if re.fullmatch(pattern, old_key):
                return None
        else:
            new_key, n_replace = re.subn(pattern, replacement, old_key)
            # Early exit of the loop
            if n_replace > 0:
                return new_key

    raise ValueError(f"Key: {old_key} could not be mapped (check the mapping).")


def convert_state_dict(original_state_dict: dict, config: TeleChat2Config):
    new_dict = {}

    head_dim = config.hidden_size // config.num_attention_heads

    for old_key, value in original_state_dict.items():
        new_key = map_old_key_to_new(old_key)
        if new_key is None:
            continue

        if "key_value." in new_key:
            k_proj, v_proj = (
                value[:head_dim, ...],
                value[head_dim : 2 * head_dim, ...],
            )
            new_dict[new_key.replace("key_value.", "k_proj.")] = k_proj
            new_dict[new_key.replace("key_value.", "v_proj.")] = v_proj
        else:
            new_dict[new_key] = value
    return new_dict


def convert_config(original_config: dict):
    key_mapping = {
        "intermediate_size": "ffn_hidden_size",
        "rms_norm_eps": "layer_norm_epsilon",
        "num_hidden_layers": "n_layer",
        "num_attention_heads": "n_head",
    }
    similar_keys_to_keep = [
        "max_position_embeddings",
        "hidden_size",
        "num_key_value_heads",
        "head_dim",
        "attention_dropout",
        "use_cache",
        "eos_token_id",
        "pad_token_id",
        "tie_word_embeddings",
        "vocab_size",
    ]
    new_config_kwargs = {k: original_config[v] for k, v in key_mapping.items()}
    new_config_kwargs.update({k: v for k, v in original_config.items() if k in similar_keys_to_keep})

    new_config = TeleChat2Config(**new_config_kwargs)
    return new_config


def convert_telechat2_model(input_dir, output_dir, use_post_processor=False):
    # Load and convert config
    with open(os.path.join(input_dir, "config.json")) as f:
        original_config = json.load(f)
    config = convert_config(original_config)
    config.save_pretrained(output_dir)

    # Load and convert weights
    original_state_dict = load_weights(input_dir)
    new_dict = convert_state_dict(original_state_dict, config)
    with torch.device("meta"):
        model = TeleChat2ForCausalLM(config)
    model.load_state_dict(new_dict, strict=True, assign=True)
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=str,
        help="Location of the local folder copied from the Hub.",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--use_post_processor",
        action="store_true",
        help="Whether to apply post processor with special tokens",
    )

    args = parser.parse_args()
    convert_telechat2_model(args.input_dir, args.output_dir, args.use_post_processor)
