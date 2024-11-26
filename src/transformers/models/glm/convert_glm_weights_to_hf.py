import argparse
import json
import os
import re

import torch
from safetensors.torch import load_file
from tokenizers import processors

from transformers import GlmConfig, GlmForCausalLM, PreTrainedTokenizerFast


# fmt: off
# `None` means we drop the key
STATE_DICT_MAPPING = {
    # CausalLM keys
    r"transformer.output_layer.weight":                                               r"lm_head.weight",

    # Model keys
    r"transformer.embedding.word_embeddings.weight":                                  r"model.embed_tokens.weight",
    r"transformer.rotary_pos_emb.inv_freq":                                           None,
    r"transformer.encoder.final_layernorm.weight":                                    r"model.norm.weight",

    # Layers keys
    r"transformer.encoder.layers.(\d+).input_layernorm.weight":                       r"model.layers.\1.input_layernorm.weight",
    r"transformer.encoder.layers.(\d+).post_attention_layernorm.weight":              r"model.layers.\1.post_attention_layernorm.weight",

    # Attention keys
    r"transformer.encoder.layers.(\d+).self_attention.dense.weight":                  r"model.layers.\1.self_attn.o_proj.weight",
    # qkv_proj will later be split in q|k|v|_proj
    r"transformer.encoder.layers.(\d+).self_attention.query_key_value.(weight|bias)": r"model.layers.\1.self_attn.qkv_proj.\2",

    # MLP keys
    r"transformer.encoder.layers.(\d+).mlp.dense_h_to_4h.weight":                     r"model.layers.\1.mlp.gate_up_proj.weight",
    r"transformer.encoder.layers.(\d+).mlp.dense_4h_to_h.weight":                     r"model.layers.\1.mlp.down_proj.weight",
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


def convert_state_dict(original_state_dict: dict, config: GlmConfig):
    new_dict = {}

    head_dim = config.hidden_size // config.num_attention_heads
    query_size = config.num_attention_heads * head_dim
    kv_size = config.num_key_value_heads * head_dim

    for old_key, value in original_state_dict.items():
        new_key = map_old_key_to_new(old_key)
        if new_key is None:
            continue

        if "qkv_proj." in new_key:
            q_proj, k_proj, v_proj = (
                value[:query_size, ...],
                value[query_size : query_size + kv_size, ...],
                value[query_size + kv_size :, ...],
            )
            new_dict[new_key.replace("qkv_proj.", "q_proj.")] = q_proj
            new_dict[new_key.replace("qkv_proj.", "k_proj.")] = k_proj
            new_dict[new_key.replace("qkv_proj.", "v_proj.")] = v_proj
        else:
            new_dict[new_key] = value
    return new_dict


def convert_config(original_config: dict):
    key_mapping = {
        "vocab_size": "padded_vocab_size",
        "intermediate_size": "ffn_hidden_size",
        "num_hidden_layers": "num_layers",
        "max_position_embeddings": "seq_length",
        "rms_norm_eps": "layernorm_epsilon",
        "head_dim": "kv_channels",
        "attention_bias": "add_qkv_bias",
    }
    similar_keys_to_keep = [
        "num_attention_heads",
        "hidden_size",
        "attention_dropout",
        "use_cache",
        "eos_token_id",
        "pad_token_id",
        "tie_word_embeddings",
    ]
    new_config_kwargs = {k: original_config[v] for k, v in key_mapping.items()}
    new_config_kwargs.update({k: v for k, v in original_config.items() if k in similar_keys_to_keep})
    new_config_kwargs["num_key_value_heads"] = (
        new_config_kwargs["num_attention_heads"]
        if not original_config["multi_query_attention"]
        else original_config["multi_query_group_num"]
    )
    new_config_kwargs["rope_theta"] = 10000.0 * getattr(original_config, "rope_ratio", 1)

    new_config = GlmConfig(**new_config_kwargs)
    return new_config


def convert_glm_tokenizer(input_dir, use_post_processor=False):
    fast_tok = PreTrainedTokenizerFast.from_pretrained(input_dir, model_input_names=["input_ids", "attention_mask"])
    if use_post_processor:
        fast_tok._tokenizer.post_processor = processors.Sequence(
            [
                processors.ByteLevel(trim_offsets=False),
                processors.TemplateProcessing(
                    single="[gMASK]:0 <sop>:0 $A:0",
                    pair="[gMASK]:0 <sop>:0 $A:0 $B:1",
                    special_tokens=[("[gMASK]", 151331), ("<sop>", 151333)],
                ),
            ],
        )
    else:
        fast_tok._tokenizer.post_processor = processors.Sequence(
            [processors.ByteLevel(trim_offsets=False)],
        )
    return fast_tok


def convert_glm_model(input_dir, output_dir, use_post_processor=False):
    # Load and convert config
    with open(os.path.join(input_dir, "config.json")) as f:
        original_config = json.load(f)
    config = convert_config(original_config)
    config.save_pretrained(output_dir)

    # Load and convert weights
    original_state_dict = load_weights(input_dir)
    new_dict = convert_state_dict(original_state_dict, config)
    with torch.device("meta"):
        model = GlmForCausalLM(config)
    model.load_state_dict(new_dict, strict=True, assign=True)
    model.save_pretrained(output_dir)

    # Load and convert tokenizer
    tokenizer = convert_glm_tokenizer(input_dir, use_post_processor)
    tokenizer.save_pretrained(output_dir)


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
    convert_glm_model(args.input_dir, args.output_dir, args.use_post_processor)
