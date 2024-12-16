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

import argparse
import gc
import json
import math
import os
from typing import List, Optional

import regex as re
import torch
import torch.nn.functional as F

from transformers import (
    GenerationConfig,
    MllamaConfig,
    MllamaForConditionalGeneration,
    MllamaImageProcessor,
    PreTrainedTokenizerFast,
)
from transformers.convert_slow_tokenizer import TikTokenConverter
from transformers.models.mllama.configuration_mllama import MllamaTextConfig, MllamaVisionConfig
from transformers.models.mllama.image_processing_mllama import get_all_supported_aspect_ratios


# fmt: off
# If a weight needs to be split in two or more keys, use `|` to indicate it. ex:
# r"text_model.layers.(\d+).attention.wqkv.weight": r"language_model.model.layers.\1.self_attn.q|k|v|_proj.weight"
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    r"text_model.norm.weight":                                                                  r"language_model.model.norm.weight",
    r"text_model.output.weight":                                                                r"language_model.lm_head.weight",
    r"text_model.tok_embeddings":                                                               r"language_model.model.embed_tokens",
    r"text_model.learnable_embedding":                                                          r"language_model.model.learnable_embedding",
    r"text_model.rope.freqs":                                                                   None, # meaning we skip it and don't want it
    # For every cross attention layer, the layer needs to be updated
    r"text_model.cross_attention_layers.(\d+).gate_attn":                                       r"language_model.model.layers.\1.cross_attn_attn_gate",
    r"text_model.cross_attention_layers.(\d+).gate_ffwd":                                       r"language_model.model.layers.\1.cross_attn_mlp_gate",
    # special key, wqkv needs to be split afterwards
    r"text_model.cross_attention_layers.(\d+).attention.w(q|k|v|o)":                            r"language_model.model.layers.\1.cross_attn.\2_proj",
    r"text_model.cross_attention_layers.(\d+).attention.(q|k)_norm":                            r"language_model.model.layers.\1.cross_attn.\2_norm",
    r"text_model.cross_attention_layers.(\d+).attention_norm.weight":                           r"language_model.model.layers.\1.input_layernorm.weight",
    r"text_model.cross_attention_layers.(\d+).attention.wk.layer_norm_weight":                  r"language_model.model.layers.\1.post_attention_layernorm.weight",
    r"text_model.cross_attention_layers.(\d+).feed_forward.w1.weight":                          r"language_model.model.layers.\1.mlp.gate_proj.weight",
    r"text_model.cross_attention_layers.(\d+).feed_forward.w2.weight":                          r"language_model.model.layers.\1.mlp.down_proj.weight",
    r"text_model.cross_attention_layers.(\d+).feed_forward.w3.weight":                          r"language_model.model.layers.\1.mlp.up_proj.weight",
    r"text_model.cross_attention_layers.(\d+).ffn_norm.weight":                                 r"language_model.model.layers.\1.post_attention_layernorm.weight",
    # self attention layers
    r"text_model.layers.(\d+).attention.w(q|k|v|o).weight":                                     r"language_model.model.layers.\1.self_attn.\2_proj.weight",
    r"text_model.layers.(\d+).attention_norm.weight":                                           r"language_model.model.layers.\1.input_layernorm.weight",
    r"text_model.layers.(\d+).feed_forward.w1.":                                                r"language_model.model.layers.\1.mlp.gate_proj.",
    r"text_model.layers.(\d+).feed_forward.w2.":                                                r"language_model.model.layers.\1.mlp.down_proj.",
    r"text_model.layers.(\d+).feed_forward.w3.":                                                r"language_model.model.layers.\1.mlp.up_proj.",
    r"text_model.layers.(\d+).ffn_norm.weight":                                                 r"language_model.model.layers.\1.post_attention_layernorm.weight",
    # Vision encoder mapping
    r"vision_model.vision_encoder.conv1._linear":                                               r"vision_model.patch_embedding",
    r'vision_model.vision_projection.':                                                         r"multi_modal_projector.",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).attn.wq":    r"vision_model.\1.layers.\2.self_attn.q_proj",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).attn.wk":    r"vision_model.\1.layers.\2.self_attn.k_proj",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).attn.wv":    r"vision_model.\1.layers.\2.self_attn.v_proj",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).attn.wo":    r"vision_model.\1.layers.\2.self_attn.o_proj",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).mlp.c_fc":   r"vision_model.\1.layers.\2.mlp.fc1",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).mlp.c_proj": r"vision_model.\1.layers.\2.mlp.fc2",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).ln_1":       r"vision_model.\1.layers.\2.input_layernorm",
    r"vision_model.vision_encoder.(global_transformer|transformer).resblocks.(\d+).ln_2":       r"vision_model.\1.layers.\2.post_attention_layernorm",
    r"vision_model.vision_encoder.global_transformer.resblocks.(\d+).(gate_ffn|gate_attn)":     r"vision_model.global_transformer.layers.\1.\2",
    r'vision_model.vision_encoder.ln_(pre|post).(weight|bias)':                                 r'vision_model.vision_encoder.layernorm_\1.\2',
    r'vision_model.vision_encoder.positional_embedding\b':                                      r'vision_model.gated_positional_embedding.embedding',
    r'vision_model.vision_encoder.gated_positional_embedding\b':                                r'vision_model.gated_positional_embedding.tile_embedding.weight',
    r'vision_model.vision_encoder.gated_positional_embedding_gate':                             r'vision_model.gated_positional_embedding.gate',
    r"vision_model.vision_encoder.pre_tile_pos_embed.embedding":                                r"vision_model.pre_tile_positional_embedding.embedding.weight",
    r"vision_model.vision_encoder.post_tile_pos_embed.embedding":                               r"vision_model.post_tile_positional_embedding.embedding.weight",
    r"vision_model.vision_encoder.pre_tile_pos_embed.gate":                                     r"vision_model.pre_tile_positional_embedding.gate",
    r"vision_model.vision_encoder.post_tile_pos_embed.gate":                                    r"vision_model.post_tile_positional_embedding.gate",
    r"vision_model.vision_encoder.(?=\w)":                                                      r"vision_model.",
}
# fmt: on

CONTEXT_LENGTH = 131072


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


def permute_for_rope(input_tensor, n_heads, dim1, dim2):
    """
    When you go from the complex ROPE formulation to sin and cos one, you need
    to permute the query and key weights (to avoid doing it on the fly)
    """
    input_tensor = input_tensor.reshape(dim1, dim2)
    input_tensor = input_tensor.view(n_heads, dim1 // n_heads // 2, 2, dim2)
    input_tensor = input_tensor.transpose(1, 2).reshape(dim1, dim2)
    return input_tensor


def pre_compute_positional_embedding(embedding):
    """
    Instead of iterating of the batch of images, and the ratios inside, we pre-compute the
    positional embeddings depending on the aspect ratio id. This is done to support `torch.compile`
    and efficient inference / training with different aspect ratios.
    """
    max_num_tiles, *shapes = embedding.shape
    hidden_size = shapes[-1]
    supported_aspect_ratios = get_all_supported_aspect_ratios(max_num_tiles)
    max_aspect_ratio_id = len(supported_aspect_ratios)  # we keep 0 index for padding
    # tile embedding does not have patches
    num_patches = 1 if len(shapes) == 2 else shapes[1]
    precomputed_embeddings = torch.zeros(
        max_aspect_ratio_id + 1,
        max_num_tiles,
        num_patches,
        hidden_size,
        device=embedding.device,
        dtype=embedding.dtype,
    )

    for i, (height, width) in enumerate(supported_aspect_ratios):
        aspect_ratio_id = i + 1  # we keep 0 index for padding
        current_embedding = embedding[:height, :width].reshape(height * width, num_patches, hidden_size)
        precomputed_embeddings[aspect_ratio_id, : height * width] = current_embedding
    precomputed_embeddings = precomputed_embeddings.flatten(1)
    return precomputed_embeddings


def is_param_different_across_shards(key):
    """
    Return `True` if the parameter is different across checkpoint shards
    and needs to be concatenated.
    """
    patterns = [r"vision_model.patch_embedding.weight",r"vision_model.(transformer|global_transformer).layers.(\d+).self_attn.(q|k|v|o)_proj.weight",r"vision_model.(transformer|global_transformer).layers.(\d+).mlp.fc1.(weight|bias)",r"vision_model.(transformer|global_transformer).layers.(\d+).mlp.fc2.weight",  r"multi_modal_projector.(weight|bias)",r"language_model.model.embed_tokens.weight",r"language_model.lm_head.weight",r"language_model.model.layers.(\d+).self_attn.(q|k|v|o)_proj.weight",r"language_model.model.layers.(\d+).cross_attn.(q|k|v|o)_proj.weight",r"language_model.model.layers.(\d+).mlp.(up|down|gate)_proj.weight",r"language_model.model.learnable_embedding.weight"]  # fmt: skip
    return any(re.search(pattern, key) for pattern in patterns)


def get_concat_dim(key):
    """
    Return the dimension to concatenate the weights on.
    """
    concat_dim_1 = [r"vision_model.(transformer|global_transformer).layers.(\d+).mlp.fc2.weight",r"vision_model.(transformer|global_transformer).layers.(\d+).self_attn.o_proj.weight",r"language_model.model.layers.(\d+).cross_attn.o_proj.weight",r"language_model.model.layers.(\d+).self_attn.o_proj.weight",r"language_model.model.layers.(\d+).mlp.down_proj.weight"]  # fmt: off
    if any(re.search(pattern, key) for pattern in concat_dim_1):
        return 1
    return 0


def compute_intermediate_size(hidden_dim, multiple_of=1024, ffn_dim_multiplier=1.3):
    hidden_dim = 4 * int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


def interpolate_positional_embedding(
    embeddings: torch.Tensor, vision_tile_size: int, vision_patch_size: int
) -> torch.Tensor:
    """
    This method allows to interpolate the pre-trained position embeddings, to be able to use the model on higher resolution
    images.
    """
    cls_embedding, positional_embedding = embeddings[:1], embeddings[1:]
    total_num_patches, dim = positional_embedding.shape

    # compute current and target number of patches for height and width
    num_patches = int(round(total_num_patches**0.5))
    new_num_patches = vision_tile_size // vision_patch_size

    # Check if the number of patches is already the desired size
    if num_patches == new_num_patches:
        return embeddings

    positional_embedding = positional_embedding.transpose(0, 1)
    positional_embedding = positional_embedding.reshape(1, dim, num_patches, num_patches)
    positional_embedding = F.interpolate(
        positional_embedding,
        size=(new_num_patches, new_num_patches),
        mode="bicubic",
        align_corners=False,
    )
    positional_embedding = positional_embedding.reshape(dim, -1).transpose(0, 1)

    embeddings = torch.cat([cls_embedding, positional_embedding], dim=0)
    return embeddings


def write_model(
    model_path,
    input_base_path,
    num_shards,
    safe_serialization=True,
    instruct=False,
):
    os.makedirs(model_path, exist_ok=True)

    with open(os.path.join(input_base_path, "params.json"), "r") as f:
        params = json.load(f)

    params = params.get("model", params)
    torch_dtype = "bfloat16"

    # ------------------------------------------------------------
    # Text model params and config
    # ------------------------------------------------------------

    # params from config
    text_vocab_size = params["vocab_size"]
    text_num_layers = params["n_layers"]
    text_dim = params["dim"]
    text_num_heads = params["n_heads"]
    text_rms_norm_eps = params["norm_eps"]
    text_rope_theta = params["rope_theta"]
    cross_attention_num_layers = params["vision_num_cross_attention_layers"]

    # some constans from original code
    rope_scaling = {
        "rope_type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    }
    max_position_embeddings = CONTEXT_LENGTH

    # compute additional params for weight conversion
    text_num_heads_per_shard = text_num_heads // num_shards
    text_dim_per_head = text_dim // text_num_heads
    text_intermediate_size = compute_intermediate_size(text_dim, multiple_of=params["multiple_of"])

    if params.get("n_kv_heads", None) is not None:
        text_num_key_value_heads = params["n_kv_heads"]  # for GQA / MQA
        text_num_key_value_heads_per_shard = text_num_key_value_heads // num_shards
        text_key_value_dim = text_dim_per_head * text_num_key_value_heads
    else:  # compatibility with other checkpoints
        text_num_key_value_heads = text_num_heads
        text_num_key_value_heads_per_shard = text_num_heads_per_shard
        text_key_value_dim = text_dim

    # cross-attention layers: 20 for 90B, 8 for 11B
    cross_attention_frequency = math.ceil(text_num_layers / cross_attention_num_layers)
    text_num_total_layers = text_num_layers + cross_attention_num_layers
    cross_attention_layers_shift = list(
        range(cross_attention_frequency - 1, text_num_total_layers, cross_attention_frequency + 1)
    )
    self_attention_layers_shift = [k for k in range(text_num_total_layers) if k not in cross_attention_layers_shift]

    bos_token_id = 128000
    eos_token_id = [128001, 128008, 128009] if instruct else 128001
    pad_token_id = 128004

    text_config = MllamaTextConfig(
        num_attention_heads=text_num_heads,
        vocab_size=text_vocab_size,
        hidden_size=text_dim,
        rms_norm_eps=text_rms_norm_eps,
        rope_theta=text_rope_theta,
        num_hidden_layers=text_num_total_layers,
        cross_attention_layers=cross_attention_layers_shift,
        intermediate_size=text_intermediate_size,
        max_position_embeddings=max_position_embeddings,
        rope_scaling=rope_scaling,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        tie_word_embeddings=False,  # Constant set to False
        torch_dtype=torch_dtype,
    )

    # ------------------------------------------------------------
    # Vision model params and config
    # ------------------------------------------------------------

    # params from config
    vision_tile_size = params["vision_chunk_size"]
    vision_max_num_tiles = params["vision_max_num_chunks"]

    # some constants from original code
    vision_patch_size = 14
    vision_num_channels = 3
    vision_num_layers = 32
    vision_num_layers_global = 8
    vision_dim = 1280
    vision_num_heads = 16
    vision_intermediate_layers_indices = [3, 7, 15, 23, 30]

    # compute additional params for weight conversion
    vision_dim_per_head = vision_dim // vision_num_heads
    vision_num_heads_per_shard = vision_num_heads // num_shards
    vision_intermediate_size = vision_dim * 4
    vision_supported_aspect_ratios = get_all_supported_aspect_ratios(vision_max_num_tiles)

    vision_config = MllamaVisionConfig(
        hidden_size=vision_dim,
        patch_size=vision_patch_size,
        num_channels=vision_num_channels,
        intermediate_size=vision_intermediate_size,
        num_hidden_layers=vision_num_layers,
        num_attention_heads=vision_num_heads,
        num_global_layers=vision_num_layers_global,
        intermediate_layers_indices=vision_intermediate_layers_indices,
        image_size=vision_tile_size,
        max_num_tiles=vision_max_num_tiles,
        supported_aspect_ratios=vision_supported_aspect_ratios,
        torch_dtype=torch_dtype,
    )

    # save config
    config = MllamaConfig(vision_config=vision_config, text_config=text_config, torch_dtype=torch_dtype)
    config.architectures = ["MllamaForConditionalGeneration"]
    config.save_pretrained(model_path)
    print("Model config saved successfully...")

    # ------------------------------------------------------------
    # Convert weights
    # ------------------------------------------------------------

    print(f"Fetching all parameters from the checkpoint at {input_base_path}...")
    if num_shards == 1:
        if os.path.exists(os.path.join(input_base_path, "consolidated.00.pth")):
            path = os.path.join(input_base_path, "consolidated.00.pth")
        else:
            path = os.path.join(input_base_path, "consolidated.pth")
        loaded = [torch.load(path, map_location="cpu", mmap=True)]
    else:
        loaded = [
            torch.load(os.path.join(input_base_path, f"consolidated.{i:02d}.pth"), map_location="cpu", mmap=True)
            for i in range(num_shards)
        ]

    print("Converting model...")
    all_keys = list(loaded[0].keys())
    new_keys = convert_old_keys_to_new_keys(all_keys)

    state_dict = {}
    for key in all_keys:
        new_key = new_keys[key]

        # In the original model, self-attention layers and cross-attention layers are different lists of layers.
        # In the converted model, they are merged into one list with corresponding index shift to preserve the order.
        if ("cross_attention" in key or "text_model.layers" in key) and "language_model" in new_key:
            shift = cross_attention_layers_shift if "cross_attention" in key else self_attention_layers_shift
            new_key = re.sub(r"layers.(\d+).", lambda _match: f"layers.{shift[int(_match.groups()[0])]}.", new_key)

        current_parameter = [chunk.pop(key).contiguous().clone() for chunk in loaded]
        if not is_param_different_across_shards(new_key):
            current_parameter = current_parameter[0]

        concat_dim = get_concat_dim(new_key)

        # Post-process the current_parameter.
        if re.search("(k|v|q)_proj.weight", new_key) and "language_model" in new_key:
            if "q_proj" in new_key:
                param_num_heads = text_num_heads
                param_num_head_per_shard = text_num_heads_per_shard
                param_dim = text_dim
            else:
                param_num_heads = text_num_key_value_heads
                param_num_head_per_shard = text_num_key_value_heads_per_shard
                param_dim = text_key_value_dim
            shards = [param.view(param_num_head_per_shard, text_dim_per_head, text_dim) for param in current_parameter]
            current_parameter = torch.cat(shards, dim=concat_dim)
            if "cross_attn" not in new_key and "v_proj.weight" not in new_key:
                current_parameter = permute_for_rope(current_parameter, param_num_heads, param_dim, text_dim)
            state_dict[new_key] = current_parameter.reshape(param_num_heads * text_dim_per_head, text_dim)

        elif "vision_model" in new_key and re.search("(k|v|q)_proj", new_key):
            shards = [
                param.view(vision_num_heads_per_shard, vision_dim_per_head, vision_dim) for param in current_parameter
            ]
            param = torch.cat(shards, dim=concat_dim)
            state_dict[new_key] = param.reshape(vision_num_heads * vision_dim_per_head, vision_dim)

        elif new_key == "vision_model.patch_embedding.weight":
            current_parameter = torch.cat(current_parameter, dim=concat_dim)
            state_dict[new_key] = current_parameter.reshape(
                -1, vision_num_channels, vision_patch_size, vision_patch_size
            )

        elif new_key.endswith("gate"):
            state_dict[new_key] = current_parameter[0].view(1)

        elif "vision_model.gated_positional_embedding.embedding" in new_key:
            current_parameter = interpolate_positional_embedding(
                current_parameter, vision_tile_size, vision_patch_size
            )
            state_dict[new_key] = current_parameter

        elif "vision_model.gated_positional_embedding.tile_embedding.weight" in new_key:
            current_parameter = current_parameter.permute(2, 0, 1, 3).flatten(1)
            current_parameter = interpolate_positional_embedding(
                current_parameter, vision_tile_size, vision_patch_size
            )
            current_parameter = current_parameter.reshape(
                -1, vision_max_num_tiles, vision_max_num_tiles, vision_dim
            ).permute(1, 2, 0, 3)
            state_dict[new_key] = pre_compute_positional_embedding(current_parameter)

        elif "tile_positional_embedding.embedding" in new_key:
            state_dict[new_key] = pre_compute_positional_embedding(current_parameter)

        elif new_key != "":
            if isinstance(current_parameter, list):
                current_parameter = torch.cat(current_parameter, dim=concat_dim)
            state_dict[new_key] = current_parameter

    state_dict["language_model.model.embed_tokens.weight"] = torch.cat(
        [
            state_dict["language_model.model.embed_tokens.weight"],
            state_dict.pop("language_model.model.learnable_embedding.weight"),
        ],
        dim=0,
    )
    del loaded
    gc.collect()

    print("Loading the checkpoint in a Mllama model.")
    with torch.device("meta"):
        model = MllamaForConditionalGeneration(config)
    model.load_state_dict(state_dict, strict=True, assign=True)
    print("Checkpoint loaded successfully.")
    del model.config._name_or_path

    print("Saving the model.")
    model.save_pretrained(model_path, safe_serialization=safe_serialization)
    del state_dict, model

    # Safety check: reload the converted model
    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    MllamaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    print("Model reloaded successfully.")

    # generation config
    if instruct:
        print("Saving generation config...")
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        generation_config.save_pretrained(model_path)


class MllamaConverter(TikTokenConverter):
    def __init__(
        self,
        vocab_file,
        special_tokens: List[str],
        pattern: str,
        model_max_length: int,
        chat_template: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(vocab_file, pattern=pattern)
        self.additional_special_tokens = special_tokens
        tokenizer = self.converted()
        if chat_template is not None:
            kwargs["chat_template"] = chat_template
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            model_input_names=["input_ids", "attention_mask"],
            model_max_length=model_max_length,
            **kwargs,
        )


def write_tokenizer(tokenizer_path: str, save_dir: str, instruct: bool = False):
    model_max_length = CONTEXT_LENGTH
    pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: W605

    # Special tokens
    num_reserved_special_tokens = 256
    special_tokens = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|reserved_special_token_0|>",
        "<|reserved_special_token_1|>",
        "<|finetune_right_pad_id|>",
        "<|step_id|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eom_id|>",  # end of message
        "<|eot_id|>",  # end of turn
        "<|python_tag|>",
    ]
    special_tokens += [
        f"<|reserved_special_token_{i + 2}|>" for i in range(num_reserved_special_tokens - len(special_tokens))
    ]
    # original tokenizer has <|image|> with 128011 token_id,
    # however, later in the code it is replaced with 128256 token_id
    special_tokens.append("<|image|>")

    # Chat template
    chat_template = (
        "{% for message in messages %}"
        "{% if loop.index0 == 0 %}"
        "{{ bos_token }}"
        "{% endif %}"
        "{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' }}"
        "{% if message['content'] is string %}"
        "{{ message['content'] }}"
        "{% else %}"
        "{% for content in message['content'] %}"
        "{% if content['type'] == 'image' %}"
        "{{ '<|image|>' }}"
        "{% elif content['type'] == 'text' %}"
        "{{ content['text'] }}"
        "{% endif %}"
        "{% endfor %}"
        "{% endif %}"
        "{{ '<|eot_id|>' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        "{% endif %}"
    )

    converter = MllamaConverter(
        vocab_file=tokenizer_path,
        pattern=pattern,
        special_tokens=special_tokens,
        model_max_length=model_max_length,
        chat_template=chat_template if instruct else None,
        bos_token="<|begin_of_text|>",
        eos_token="<|end_of_text|>" if not instruct else "<|eot_id|>",
        pad_token="<|finetune_right_pad_id|>",
    )
    tokenizer = converter.tokenizer
    tokenizer.save_pretrained(save_dir)

    if instruct:
        print("Saving chat template...")
        chat_template_path = os.path.join(save_dir, "chat_template.json")
        with open(chat_template_path, "w") as f:
            json.dump({"chat_template": chat_template}, f, indent=2)


def write_image_processor(config_path: str, save_dir: str):
    with open(config_path, "r") as f:
        params = json.load(f)

    tile_size = params["vision_chunk_size"]
    max_image_tiles = params["vision_max_num_chunks"]

    image_processor = MllamaImageProcessor(
        do_resize=True,
        size={"height": tile_size, "width": tile_size},
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
        do_pad=True,
        max_image_tiles=max_image_tiles,
    )

    image_processor.save_pretrained(save_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="Llama-3.2-11B-Vision/original",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--output_dir",
        default="Llama-3.2-11B-Vision",
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--safe_serialization", default=True, type=bool, help="Whether or not to save using `safetensors`."
    )
    parser.add_argument(
        "--special_tokens",
        default=None,
        type=List[str],
        help="The list of special tokens that should be added to the model.",
    )
    parser.add_argument(
        "--num_shards",
        default=1,
        type=int,
        help="The number of individual shards used for the model. Does not have to be the same as the number of consolidated_xx.pth",
    )
    parser.add_argument(
        "--instruct",
        action="store_true",
        help="Whether the model is an instruct model",
    )
    args = parser.parse_args()
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        safe_serialization=args.safe_serialization,
        num_shards=args.num_shards,
        instruct=args.instruct,
    )

    write_tokenizer(
        tokenizer_path=os.path.join(args.input_dir, "tokenizer.model"),
        save_dir=args.output_dir,
        instruct=args.instruct,
    )

    write_image_processor(
        config_path=os.path.join(args.input_dir, "params.json"),
        save_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
