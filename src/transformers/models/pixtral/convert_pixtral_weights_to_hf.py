# coding=utf-8
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
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
import json
import os

import regex as re
import torch
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from safetensors.torch import load_file as safe_load_file
from tokenizers import Regex, Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE

from transformers import (
    LlavaConfig,
    LlavaForConditionalGeneration,
    MistralConfig,
    PixtralImageProcessor,
    PixtralProcessor,
    PixtralVisionConfig,
)
from transformers.convert_slow_tokenizer import bytes_to_unicode


"""
# Here is how to get the original tokens!
model_name = "mistralai/Pixtral-12B-2409"
tok = MistralTokenizer.from_model(model_name)

from mistral_common.protocol.instruct.request import ChatCompletionRequest, UserMessage, ImageChunk, TextChunk

EXPECTED_TOKENS = tok.encode_chat_completion(
    ChatCompletionRequest(
        messages=[
            UserMessage(
                content=[
                    TextChunk(text="Describe the images"),
                ] + [ImageChunk(image=img) for img in IMG_URLS]
            )
        ],
        model="pixtral",
    )
)
assert tokenizer.decode(inputs["input_ids"][0]) == EXPECTED_TOKENS
"""

OLD_KEY_TO_NEW_KEY_MAPPING = {
    # Layer Normalization Weights
    r"vision_encoder.transformer.layers.(\d+).input_layernorm.weight": r"vision_tower.transformer.layers.\1.attention_norm.weight",
    r"vision_encoder.transformer.layers.(\d+).ffn_norm.weight": r"vision_tower.transformer.layers.\1.ffn_norm.weight",
    # Self Attention Projections
    r"vision_encoder.transformer.layers.(\d+).attention.wq.weight": r"vision_tower.transformer.layers.\1.attention.q_proj.weight",
    r"vision_encoder.transformer.layers.(\d+).attention.wk.weight": r"vision_tower.transformer.layers.\1.attention.k_proj.weight",
    r"vision_encoder.transformer.layers.(\d+).attention.wv.weight": r"vision_tower.transformer.layers.\1.attention.v_proj.weight",
    r"vision_encoder.transformer.layers.(\d+).attention.wo.weight": r"vision_tower.transformer.layers.\1.attention.o_proj.weight",
    # MLP Projections
    r"vision_encoder.transformer.layers.(\d+).feed_forward.w1.weight": r"vision_tower.transformer.layers.\1.feed_forward.gate_proj.weight",
    r"vision_encoder.transformer.layers.(\d+).feed_forward.w2.weight": r"vision_tower.transformer.layers.\1.feed_forward.down_proj.weight",
    r"vision_encoder.transformer.layers.(\d+).feed_forward.w3.weight": r"vision_tower.transformer.layers.\1.feed_forward.up_proj.weight",
    # Additional mappings
    r"vision_encoder": r"vision_tower",
    r"vision_language_adapter.w_in": r"multi_modal_projector.linear_1",
    r"vision_language_adapter.w_out": r"multi_modal_projector.linear_2",
    r"layers.(\d+).attention.wq.weight": r"language_model.model.layers.\1.self_attn.q_proj.weight",
    r"layers.(\d+).attention.wk.weight": r"language_model.model.layers.\1.self_attn.k_proj.weight",
    r"layers.(\d+).attention.wv.weight": r"language_model.model.layers.\1.self_attn.v_proj.weight",
    r"layers.(\d+).attention.wo.weight": r"language_model.model.layers.\1.self_attn.o_proj.weight",
    r"layers.(\d+).feed_forward.w1.weight": r"language_model.model.layers.\1.mlp.gate_proj.weight",
    r"layers.(\d+).feed_forward.w2.weight": r"language_model.model.layers.\1.mlp.down_proj.weight",
    r"layers.(\d+).feed_forward.w3.weight": r"language_model.model.layers.\1.mlp.up_proj.weight",
    r"layers.(\d+).ffn_norm.weight": r"language_model.model.layers.\1.post_attention_layernorm.weight",
    r"layers.(\d+).attention_norm.weight": r"language_model.model.layers.\1.input_layernorm.weight",
    r"tok_embeddings.weight": r"language_model.model.embed_tokens.weight",
    r"output.weight": r"language_model.lm_head.weight",
    r"norm.weight": r"language_model.model.norm.weight",
}


class MistralConverter:
    """
    A general tiktoken converter.
    """

    def __init__(
        self,
        vocab=None,
        pattern=r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
        add_prefix_space=False,
        additional_special_tokens=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args)
        self.vocab = vocab
        self.pattern = pattern
        self.add_prefix_space = add_prefix_space
        self.additional_special_tokens = additional_special_tokens

    def extract_vocab_merges_from_model(self, vocab: str):
        bpe_ranks = vocab
        byte_encoder = bytes_to_unicode()

        def token_bytes_to_string(b):
            return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

        merges = []
        vocab = {}
        for idx, (token, rank) in enumerate(bpe_ranks.items()):
            if token not in self.additional_special_tokens:
                vocab[token_bytes_to_string(token)] = idx
                if len(token) == 1:
                    continue
                local = []
                for index in range(1, len(token)):
                    piece_l, piece_r = token[:index], token[index:]
                    if piece_l in bpe_ranks and piece_r in bpe_ranks and (piece_l + piece_r) in bpe_ranks:
                        local.append((piece_l, piece_r, rank))
                local = sorted(local, key=lambda x: (bpe_ranks[x[0]], bpe_ranks[x[1]]), reverse=False)
                merges.extend(local)
            else:
                vocab[token] = idx
        merges = sorted(merges, key=lambda val: val[2], reverse=False)
        merges = [(token_bytes_to_string(val[0]), token_bytes_to_string(val[1])) for val in merges]
        return vocab, merges

    def tokenizer(self):
        vocab_scores, merges = self.extract_vocab_merges_from_model(self.vocab)
        tokenizer = Tokenizer(BPE(vocab_scores, merges, fuse_unk=False))
        if hasattr(tokenizer.model, "ignore_merges"):
            tokenizer.model.ignore_merges = True
        return tokenizer

    def converted(self) -> Tokenizer:
        tokenizer = self.tokenizer()
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(self.pattern), behavior="isolated", invert=False),
                pre_tokenizers.ByteLevel(add_prefix_space=self.add_prefix_space, use_regex=False),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.add_special_tokens(self.additional_special_tokens)

        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        return tokenizer


def convert_mistral_tokenizer(model_file):
    from transformers import LlamaTokenizer

    mistral_tokenizer = MistralTokenizer.from_file(model_file)
    vocab = mistral_tokenizer.instruct_tokenizer.tokenizer.vocab()
    control_token_ids = mistral_tokenizer.instruct_tokenizer.tokenizer._control_tokens
    all_special = [vocab[id] for id in control_token_ids]
    hf_tokenizer = LlamaTokenizer(model_file)
    # Do I need to exclude tokens that are already special?
    hf_tokenizer.add_special_tokens({"additional_special_tokens": all_special})
    hf_tokenizer.model_input_names = ["input_ids", "attention_mask"]
    return hf_tokenizer


def permute_for_rope(value, n_heads, config):
    dim1 = value.shape[0]
    dim2 = config.hidden_size
    return value.view(n_heads, dim1 // n_heads // 2, 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def convert_dictionary(original_state_dict, vision_config, text_config):
    new_dict = {}

    all_keys = "\n" + "\n".join(original_state_dict.keys())
    old_keys = all_keys
    for old, new in OLD_KEY_TO_NEW_KEY_MAPPING.items():
        all_keys = re.sub(r"\n" + old, r"\n" + new, all_keys)

    OLD_TO_NEW = dict(zip(old_keys.split("\n"), all_keys.split("\n")))

    for key, value in original_state_dict.items():
        new_key = OLD_TO_NEW[key]
        if "vision_encoder" in key:
            _config = vision_config
            num_attention_heads = _config.num_attention_heads
        else:
            _config = text_config
            if "q_proj" in new_key:
                num_attention_heads = _config.num_attention_heads
            if "k_proj" in new_key:
                num_attention_heads = _config.num_key_value_heads

        if "q_proj" in new_key or "k_proj" in new_key:
            value = permute_for_rope(value, num_attention_heads, _config)

        new_dict[new_key] = value
    return new_dict


MISTRAL_CONFIG_MAPPING = {
    "dim": "hidden_size",
    "hidden_dim": "intermediate_size",
    "n_kv_heads": "num_key_value_heads",
    "n_heads": "num_attention_heads",
    "n_layers": "num_hidden_layers",
}


def convert_mistral_model(input_dir, output_dir):
    vision_config = {}
    if os.path.isfile(f"{input_dir}/params.json"):
        with open(f"{input_dir}/params.json") as f:
            param_json = json.load(f)
        vision_config = param_json.pop("vision_encoder")
        for k, v in MISTRAL_CONFIG_MAPPING.items():
            value = param_json.pop(k)
            param_json[v] = value
        if "hidden_act" not in vision_config:
            vision_config["hidden_act"] = "silu"
        text_config = MistralConfig(
            **param_json,
            hidden_act="silu",
            sliding_window=None,
            tie_word_embeddings=False,
            is_composition=True,
            rms_norm_eps=1e-5,
        )
    else:
        text_config = MistralConfig(
            attention_dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
            head_dim=128,
            hidden_act="silu",
            hidden_size=5120,
            initializer_range=0.02,
            intermediate_size=14336,
            max_position_embeddings=1024000,
            model_type="mistral",
            num_attention_heads=32,
            num_hidden_layers=40,
            num_key_value_heads=8,
            rms_norm_eps=1e-05,
            rope_theta=1000000000.0,
            sliding_window=None,
            tie_word_embeddings=False,
            vocab_size=131072,
        )
    adapter_bias = vision_config.pop("adapter_bias", True)
    vision_config = PixtralVisionConfig(**vision_config)
    config = LlavaConfig(
        vision_config,
        text_config,
        vision_feature_layer=-1,
        image_token_index=10,
        vision_feature_select_strategy="full",
        image_seq_length=1,
        multimodal_projector_bias=adapter_bias,
    )
    config.architectures = ["LlavaForConditionalGeneration"]
    config.save_pretrained(output_dir)
    full_original_state_dict = {}
    safetensors_files = sorted([file for file in os.listdir(input_dir) if file.endswith(".safetensors")])
    if len(safetensors_files) == 1:
        full_original_state_dict = safe_load_file(f"{input_dir}/consolidated.safetensors")
    else:
        for file in safetensors_files:
            loaded_dict = safe_load_file(f"{input_dir}/{file}")
            full_original_state_dict.update(loaded_dict)

    new_dict = convert_dictionary(full_original_state_dict, vision_config, text_config)
    with torch.device("meta"):
        model = LlavaForConditionalGeneration(config)
    model.load_state_dict(new_dict, strict=True, assign=True)
    model.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        help="Location to write HF model and tokenizer",
        required=True,
    )
    parser.add_argument(
        "--tokenizer_file", help="Location of the specific tokenizer model file to use.", required=True
    )
    parser.add_argument(
        "--chat_template_file",
        help="Optional file containing a raw chat template. Will be set as the processor's chat template.",
        required=False,
    )

    args = parser.parse_args()
    convert_mistral_model(args.input_dir, args.output_dir)
    tokenizer = convert_mistral_tokenizer(args.tokenizer_file)
    image_processor = PixtralImageProcessor()
    processor = PixtralProcessor(tokenizer=tokenizer, image_processor=image_processor, image_token="[IMG]")
    if args.chat_template_file:
        processor.chat_template = open(args.chat_template_file).read()
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
