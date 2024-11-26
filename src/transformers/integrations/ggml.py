# coding=utf-8
# Copyright 2024 The ggml.ai team and The HuggingFace Inc. team. and pygguf author (github.com/99991)
# https://github.com/99991/pygguf
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
"""
Integration with GGML / The file is copied and adapted from https://github.com/99991/pygguf
with extra methods beings exposed
"""

from array import array

import numpy as np
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers, processors
from tokenizers.models import BPE, Unigram

from .. import AddedToken
from ..convert_slow_tokenizer import GPT2Converter, LlamaConverter, Qwen2Converter, T5Converter
from ..utils import logging
from ..utils.logging import tqdm


logger = logging.get_logger(__name__)


GGUF_TENSOR_MAPPING = {
    "llama": {
        "token_embd": "model.embed_tokens",
        "blk": "model.layers",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
        "ffn_gate": "mlp.gate_proj",
        "ffn_norm": "post_attention_layernorm",
        "attn_norm": "input_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_v": "self_attn.v_proj",
        "attn_k": "self_attn.k_proj",
        "attn_output": "self_attn.o_proj",
        "output.weight": "lm_head.weight",
        "output_norm": "model.norm",
    },
    "mistral": {
        "token_embd": "model.embed_tokens",
        "blk": "model.layers",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
        "ffn_gate": "mlp.gate_proj",
        "ffn_norm": "post_attention_layernorm",
        "attn_norm": "input_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_v": "self_attn.v_proj",
        "attn_k": "self_attn.k_proj",
        "attn_output": "self_attn.o_proj",
        "output.weight": "lm_head.weight",
        "output_norm": "model.norm",
    },
    "qwen2": {
        "token_embd": "model.embed_tokens",
        "blk": "model.layers",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
        "ffn_gate": "mlp.gate_proj",
        "ffn_norm": "post_attention_layernorm",
        "attn_norm": "input_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_v": "self_attn.v_proj",
        "attn_k": "self_attn.k_proj",
        "attn_output": "self_attn.o_proj",
        "output.weight": "lm_head.weight",
        "output_norm": "model.norm",
    },
    "qwen2moe": {
        "token_embd": "model.embed_tokens",
        "blk": "model.layers",
        "ffn_up_exps": "mlp.experts",
        "ffn_up_shexp": "mlp.shared_expert.up_proj",
        "ffn_down_exps": "mlp.experts",
        "ffn_down_shexp": "mlp.shared_expert.down_proj",
        "ffn_norm": "post_attention_layernorm",
        "ffn_gate_inp.weight": "mlp.gate.weight",
        "ffn_gate_exps": "mlp.experts",
        "ffn_gate_shexp": "mlp.shared_expert.gate_proj",
        "ffn_gate_inp_shexp": "mlp.shared_expert_gate",
        "attn_norm": "input_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_v": "self_attn.v_proj",
        "attn_k": "self_attn.k_proj",
        "attn_output": "self_attn.o_proj",
        "output.weight": "lm_head.weight",
        "output_norm": "model.norm",
    },
    "phi3": {
        "token_embd": "model.embed_tokens",
        "blk": "model.layers",
        "ffn_up": "mlp.gate_up_proj",
        "ffn_down": "mlp.down_proj",
        "ffn_gate": "mlp.gate_up_proj",
        "ffn_norm": "post_attention_layernorm",
        "attn_norm": "input_layernorm",
        "attn_qkv": "self_attn.qkv_proj",
        "attn_output": "self_attn.o_proj",
        "output.weight": "lm_head.weight",
        "output_norm": "model.norm",
    },
    "bloom": {
        "token_embd.weight": "transformer.word_embeddings.weight",
        "token_embd_norm": "transformer.word_embeddings_layernorm",
        "blk": "transformer.h",
        "ffn_up": "mlp.dense_h_to_4h",
        "ffn_down": "mlp.dense_4h_to_h",
        "ffn_norm": "post_attention_layernorm",
        "attn_norm": "input_layernorm",
        "attn_qkv": "self_attention.query_key_value",
        "attn_output": "self_attention.dense",
        "output.weight": "lm_head.weight",
        "output_norm": "transformer.ln_f",
    },
    "falcon7b": {
        "token_embd": "word_embeddings",
        "blk": "h",
        "ffn_up": "mlp.dense_h_to_4h",
        "ffn_down": "mlp.dense_4h_to_h",
        "attn_norm": "input_layernorm",
        "attn_qkv": "self_attention.query_key_value",
        "attn_output": "self_attention.dense",
        ".output.": ".lm_head.",
        "output_norm": "ln_f",
    },
    "falcon40b": {
        "token_embd": "word_embeddings",
        "blk": "h",
        "ffn_up": "mlp.dense_h_to_4h",
        "ffn_down": "mlp.dense_4h_to_h",
        ".attn_norm.": ".ln_mlp.",
        "attn_norm_2": "ln_attn",
        "attn_qkv": "self_attention.query_key_value",
        "attn_output": "self_attention.dense",
        ".output.": ".lm_head.",
        "output_norm": "ln_f",
    },
    "t5": {
        "token_embd": "shared",
        "dec.blk.{bid}.attn_q": "decoder.block.{bid}.layer.0.SelfAttention.q",
        "dec.blk.{bid}.attn_k": "decoder.block.{bid}.layer.0.SelfAttention.k",
        "dec.blk.{bid}.attn_v": "decoder.block.{bid}.layer.0.SelfAttention.v",
        "dec.blk.{bid}.attn_o": "decoder.block.{bid}.layer.0.SelfAttention.o",
        "dec.blk.{bid}.attn_rel_b": "decoder.block.{bid}.layer.0.SelfAttention.relative_attention_bias",
        "dec.blk.{bid}.attn_norm": "decoder.block.{bid}.layer.0.layer_norm",
        "dec.blk.{bid}.cross_attn_q": "decoder.block.{bid}.layer.1.EncDecAttention.q",
        "dec.blk.{bid}.cross_attn_k": "decoder.block.{bid}.layer.1.EncDecAttention.k",
        "dec.blk.{bid}.cross_attn_v": "decoder.block.{bid}.layer.1.EncDecAttention.v",
        "dec.blk.{bid}.cross_attn_o": "decoder.block.{bid}.layer.1.EncDecAttention.o",
        "dec.blk.{bid}.cross_attn_norm": "decoder.block.{bid}.layer.1.layer_norm",
        "dec.blk.{bid}.ffn_gate": "decoder.block.{bid}.layer.2.DenseReluDense.wi_0",
        "dec.blk.{bid}.ffn_up": "decoder.block.{bid}.layer.2.DenseReluDense.wi_1",
        "dec.blk.{bid}.ffn_down": "decoder.block.{bid}.layer.2.DenseReluDense.wo",
        "dec.blk.{bid}.ffn_norm": "decoder.block.{bid}.layer.2.layer_norm",
        "dec.output_norm": "decoder.final_layer_norm",
        "enc.blk.{bid}.attn_q": "encoder.block.{bid}.layer.0.SelfAttention.q",
        "enc.blk.{bid}.attn_k": "encoder.block.{bid}.layer.0.SelfAttention.k",
        "enc.blk.{bid}.attn_v": "encoder.block.{bid}.layer.0.SelfAttention.v",
        "enc.blk.{bid}.attn_o": "encoder.block.{bid}.layer.0.SelfAttention.o",
        "enc.blk.{bid}.attn_rel_b": "encoder.block.{bid}.layer.0.SelfAttention.relative_attention_bias",
        "enc.blk.{bid}.attn_norm": "encoder.block.{bid}.layer.0.layer_norm",
        "enc.blk.{bid}.ffn_gate": "encoder.block.{bid}.layer.1.DenseReluDense.wi_0",
        "enc.blk.{bid}.ffn_up": "encoder.block.{bid}.layer.1.DenseReluDense.wi_1",
        "enc.blk.{bid}.ffn_down": "encoder.block.{bid}.layer.1.DenseReluDense.wo",
        "enc.blk.{bid}.ffn_norm": "encoder.block.{bid}.layer.1.layer_norm",
        "enc.output_norm": "encoder.final_layer_norm",
        "output.weight": "lm_head.weight",
    },
    "t5encoder": {
        "token_embd": "shared",
        "enc.blk.{bid}.attn_q": "encoder.block.{bid}.layer.0.SelfAttention.q",
        "enc.blk.{bid}.attn_k": "encoder.block.{bid}.layer.0.SelfAttention.k",
        "enc.blk.{bid}.attn_v": "encoder.block.{bid}.layer.0.SelfAttention.v",
        "enc.blk.{bid}.attn_o": "encoder.block.{bid}.layer.0.SelfAttention.o",
        "enc.blk.{bid}.attn_rel_b": "encoder.block.{bid}.layer.0.SelfAttention.relative_attention_bias",
        "enc.blk.{bid}.attn_norm": "encoder.block.{bid}.layer.0.layer_norm",
        "enc.blk.{bid}.ffn_gate": "encoder.block.{bid}.layer.1.DenseReluDense.wi_0",
        "enc.blk.{bid}.ffn_up": "encoder.block.{bid}.layer.1.DenseReluDense.wi_1",
        "enc.blk.{bid}.ffn_down": "encoder.block.{bid}.layer.1.DenseReluDense.wo",
        "enc.blk.{bid}.ffn_norm": "encoder.block.{bid}.layer.1.layer_norm",
        "enc.output_norm": "encoder.final_layer_norm",
    },
    "stablelm": {
        "token_embd": "model.embed_tokens",
        "blk": "model.layers",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
        "ffn_gate": "mlp.gate_proj",
        "ffn_norm": "post_attention_layernorm",
        "attn_norm": "input_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_v": "self_attn.v_proj",
        "attn_k": "self_attn.k_proj",
        "attn_output": "self_attn.o_proj",
        "output.weight": "lm_head.weight",
        "output_norm": "model.norm",
    },
    "gpt2": {
        "token_embd": "transformer.wte",
        "blk": "transformer.h",
        "position_embd": "transformer.wpe",
        "output_norm": "transformer.ln_f",
        "attn_norm": "ln_1",
        "attn_qkv": "attn.c_attn",
        "attn_output.weight": "attn.c_proj.weight",
        "attn_output.bias": "attn.c_proj.bias",
        "ffn_norm": "ln_2",
        "ffn_up": "mlp.c_fc",
        "ffn_down": "mlp.c_proj",
    },
    "starcoder2": {
        "token_embd": "model.embed_tokens",
        "blk": "model.layers",
        "ffn_up": "mlp.c_fc",
        "ffn_down": "mlp.c_proj",
        "ffn_norm": "post_attention_layernorm",
        "attn_norm": "input_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_v": "self_attn.v_proj",
        "attn_k": "self_attn.k_proj",
        "attn_output": "self_attn.o_proj",
        "output.weight": "lm_head.weight",
        "output_norm": "model.norm",
    },
    "mamba": {
        "token_embd": "backbone.embeddings",
        "blk": "backbone.layers",
        "ssm_a": "mixer.A_log",
        "ssm_conv1d": "mixer.conv1d",
        "ssm_in": "mixer.in_proj",
        "ssm_out": "mixer.out_proj",
        "ssm_x": "mixer.x_proj",
        "ssm_dt": "mixer.dt_proj",
        "attn_norm": "norm",
        "output_norm": "backbone.norm_f",
        "output.weight": "lm_head.weight",
    },
    "nemotron": {
        "token_embd": "model.embed_tokens",
        "blk": "model.layers",
        "ffn_up": "mlp.up_proj",
        "ffn_down": "mlp.down_proj",
        "ffn_norm": "post_attention_layernorm",
        "attn_norm": "input_layernorm",
        "attn_q": "self_attn.q_proj",
        "attn_v": "self_attn.v_proj",
        "attn_k": "self_attn.k_proj",
        "attn_output": "self_attn.o_proj",
        "output.weight": "lm_head.weight",
        "output_norm": "model.norm",
    },
}


GGUF_CONFIG_MAPPING = {
    "general": {
        "architecture": "model_type",
        "name": "_model_name_or_path",
    },
    "llama": {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        # NOTE: rope.dimension_count==head_dim only suitable for llama/mistral
        "rope.dimension_count": "head_dim",
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    },
    "mistral": {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        # NOTE: rope.dimension_count==head_dim only suitable for llama/mistral
        "rope.dimension_count": "head_dim",
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    },
    "qwen2": {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    },
    "qwen2moe": {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
        "expert_count": "num_experts",
        "expert_used_count": "num_experts_per_tok",
    },
    "falcon": {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    },
    "tokenizer": {
        "ggml.bos_token_id": "bos_token_id",
        "ggml.eos_token_id": "eos_token_id",
        "ggml.unknown_token_id": "unk_token_id",
        "ggml.padding_token_id": "pad_token_id",
    },
    "phi3": {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "rms_norm_eps",
        "vocab_size": "vocab_size",
    },
    "bloom": {
        "block_count": "n_layer",
        "embedding_length": "hidden_size",
        "attention.head_count": "n_head",
        "vocab_size": "vocab_size",
        "attention.layer_norm_epsilon": "layer_norm_epsilon",
    },
    "t5": {
        "context_length": "n_positions",
        "block_count": "num_layers",
        "feed_forward_length": "d_ff",
        "embedding_length": "d_model",
        "attention.key_length": "d_kv",
        "attention.head_count": "num_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_epsilon": "layer_norm_epsilon",
        "attention.relative_buckets_count": "relative_attention_num_buckets",
        "decoder_start_token_id": "decoder_start_token_id",
        "vocab_size": "vocab_size",
    },
    "stablelm": {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_epsilon": "layer_norm_eps",
        "vocab_size": "vocab_size",
    },
    "gpt2": {
        "block_count": "n_layer",
        "context_length": "n_ctx",
        "embedding_length": "n_embd",
        "feed_forward_length": "feed_forward_length",
        "attention.head_count": "n_head",
        "attention.layer_norm_epsilon": "layer_norm_epsilon",
    },
    "starcoder2": {
        "block_count": "num_hidden_layers",
        "context_length": "max_position_embeddings",
        "embedding_length": "hidden_size",
        "feed_forward_length": "intermediate_size",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_epsilon": "norm_epsilon",
    },
    "mamba": {
        "vocab_size": "vocab_size",
        "context_length": "max_position_embeddings",
        "embedding_length": "hidden_size",
        "attention.layer_norm_rms_epsilon": "layer_norm_epsilon",
        "block_count": "num_hidden_layers",
        "ssm.conv_kernel": "conv_kernel",
        "ssm.state_size": "state_size",
        "ssm.time_step_rank": "time_step_rank",
        "ssm.inner_size": "intermediate_size",
    },
    "nemotron": {
        "context_length": "max_position_embeddings",
        "block_count": "num_hidden_layers",
        "feed_forward_length": "intermediate_size",
        "embedding_length": "hidden_size",
        "rope.dimension_count": None,
        "rope.freq_base": "rope_theta",
        "attention.head_count": "num_attention_heads",
        "attention.head_count_kv": "num_key_value_heads",
        "attention.layer_norm_rms_epsilon": "norm_eps",
        "vocab_size": "vocab_size",
    },
}

GGUF_TOKENIZER_MAPPING = {
    "tokenizer": {
        "ggml.model": "tokenizer_type",
        "ggml.tokens": "tokens",
        "ggml.scores": "scores",
        "ggml.token_type": "token_type",
        "ggml.merges": "merges",
        "ggml.bos_token_id": "bos_token_id",
        "ggml.eos_token_id": "eos_token_id",
        "ggml.unknown_token_id": "unk_token_id",
        "ggml.padding_token_id": "pad_token_id",
        "ggml.add_space_prefix": "add_prefix_space",
    },
    "tokenizer_config": {
        "chat_template": "chat_template",
        "ggml.model": "model_type",
        "ggml.bos_token_id": "bos_token_id",
        "ggml.eos_token_id": "eos_token_id",
        "ggml.unknown_token_id": "unk_token_id",
        "ggml.padding_token_id": "pad_token_id",
    },
}


def _gguf_parse_value(_value, data_type):
    if not isinstance(data_type, list):
        data_type = [data_type]
    if len(data_type) == 1:
        data_type = data_type[0]
        array_data_type = None
    else:
        if data_type[0] != 9:
            raise ValueError("Received multiple types, therefore expected the first type to indicate an array.")
        data_type, array_data_type = data_type

    if data_type in [0, 1, 2, 3, 4, 5, 10, 11]:
        _value = int(_value[0])
    elif data_type in [6, 12]:
        _value = float(_value[0])
    elif data_type in [7]:
        _value = bool(_value[0])
    elif data_type in [8]:
        _value = array("B", list(_value)).tobytes().decode()
    elif data_type in [9]:
        _value = _gguf_parse_value(_value, array_data_type)
    return _value


class GGUFTokenizerSkeleton:
    def __init__(self, dict_):
        for k, v in dict_.items():
            setattr(self, k, v)

        if not hasattr(self, "merges"):
            if not hasattr(self, "tokens") or not hasattr(self, "scores"):
                raise ValueError(
                    "tokens and scores need to be passed for a LLaMa tokenizer without merges to be instantiated."
                )
            tokens = self.tokens
            scores = self.scores
            vocab = {t: scores[i] for i, t in enumerate(tokens)}

            logger.warning("Merges were not in checkpoint, building merges on the fly.")
            merges = []
            for merge, piece_score in tqdm(vocab.items()):
                local = []
                for index in range(1, len(merge)):
                    piece_l, piece_r = merge[:index], merge[index:]
                    if piece_l in tokens and piece_r in tokens:
                        local.append((piece_l, piece_r, piece_score))
                local = sorted(local, key=lambda x: (vocab[x[0]], vocab[x[1]]), reverse=True)
                merges.extend(local)
            merges = sorted(merges, key=lambda val: val[2], reverse=True)
            merges = [(val[0], val[1]) for val in merges]
            self.merges = merges
        else:
            self.merges = [tuple(merge.split(" ")) for merge in self.merges]
            if not hasattr(self, "scores"):
                self.scores = [None for _ in range(len(self.tokens))]

        if not hasattr(self, "added_tokens"):
            self.added_tokens = []

        if not hasattr(self, "unk_token_id"):
            self.unk_token_id = None

        # Llama2 uses the field `unknown_token_id`
        if hasattr(self, "unknown_token_id") and self.unk_token_id is None:
            self.unk_token_id = self.unknown_token_id


class GGUFLlamaConverter(LlamaConverter):
    def __init__(self, tokenizer_dict):
        self.proto = GGUFTokenizerSkeleton(tokenizer_dict)
        self.original_tokenizer = self.proto
        self.additional_kwargs = {}
        self.is_llama_3_tokenizer = getattr(self.proto, "tokenizer_type", "llama") != "llama"

    def vocab(self, proto):
        return list(zip(proto.tokens, proto.scores))

    def merges(self, proto):
        return proto.merges

    def tokenizer(self, proto):
        vocab_scores = self.vocab(self.proto)
        merges = self.merges(self.proto)
        bpe_vocab = {word: i for i, (word, _score) in enumerate(vocab_scores)}

        unk_token = proto.tokens[proto.unk_token_id] if proto.unk_token_id is not None else None
        bos_token = proto.tokens[proto.bos_token_id] if getattr(proto, "bos_token_id", None) is not None else None
        eos_token = proto.tokens[proto.bos_token_id] if getattr(proto, "eos_token_id", None) is not None else None

        tokenizer = Tokenizer(
            BPE(
                bpe_vocab,
                merges,
                unk_token=unk_token,
                fuse_unk=True,
                byte_fallback=True,
            )
        )

        special_tokens = []

        if not hasattr(self.proto, "token_type"):
            if unk_token is not None:
                special_tokens.append(AddedToken(unk_token, normalized=False, special=True))

            if bos_token is not None:
                special_tokens.append(AddedToken(bos_token, normalized=False, special=True))

            if eos_token is not None:
                special_tokens.append(AddedToken(eos_token, normalized=False, special=True))
        else:
            # 3 stands for special tokens
            special_tokens_idx = np.where(np.array(self.proto.token_type) == 3)[0]

            for idx in special_tokens_idx:
                special_tokens.append(AddedToken(self.proto.tokens[idx], normalized=False, special=True))

        if len(special_tokens) != 0:
            tokenizer.add_special_tokens(special_tokens)

        if len(self.proto.added_tokens) != 0:
            tokenizer.add_tokens(
                [AddedToken(added_token, normalized=False, special=False) for added_token in self.proto.added_tokens]
            )

        self.additional_kwargs["unk_token"] = unk_token
        self.additional_kwargs["eos_token"] = bos_token
        self.additional_kwargs["bos_token"] = eos_token

        if self.is_llama_3_tokenizer:
            self.additional_kwargs["add_prefix_space"] = None
            self.additional_kwargs["clean_up_tokenization_spaces"] = True

            self.additional_kwargs["legacy"] = False
            self.original_tokenizer.legacy = False

        return tokenizer

    def decoder(self, replacement, add_prefix_space):
        sequence = [
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Replace("▁", " "),
        ]

        if self.is_llama_3_tokenizer:
            sequence += [decoders.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=True)]

        if add_prefix_space:
            sequence += [decoders.Strip(content=" ", left=1)]
        return decoders.Sequence(sequence)

    def converted(self):
        # Copied partly from converted method in SpmConverter class
        tokenizer = self.tokenizer(self.proto)

        # Tokenizer assemble
        normalizer = self.normalizer(self.proto)
        if normalizer is not None:
            tokenizer.normalizer = normalizer

        replacement = "▁"
        add_prefix_space = True
        if hasattr(self.original_tokenizer, "add_prefix_space"):
            add_prefix_space = self.original_tokenizer.add_prefix_space

        pre_tokenizer = self.pre_tokenizer(replacement, add_prefix_space)
        if pre_tokenizer is not None:
            tokenizer.pre_tokenizer = pre_tokenizer

        tokenizer.decoder = self.decoder(replacement, add_prefix_space)
        post_processor = self.post_processor()
        if post_processor:
            tokenizer.post_processor = post_processor

        # HACK: patch the llama-3 tokenizer to use the correspinding pre-tokenizer
        # and normalizer
        if self.is_llama_3_tokenizer:
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
                add_prefix_space=False, trim_offsets=False, use_regex=True
            )
            # This is tricky as the additional kwargs are passed after legacy is force-set in LlamaTokenizer's
            # init.
            tokenizer.normalizer = normalizers.Sequence([])

        return tokenizer


class GGUFQwen2Converter(Qwen2Converter):
    def __init__(self, tokenizer_dict):
        self.original_tokenizer = GGUFTokenizerSkeleton(tokenizer_dict)
        self.additional_kwargs = {}

    def converted(self) -> Tokenizer:
        vocab = {word: i for i, word in enumerate(self.original_tokenizer.tokens)}
        merges = self.original_tokenizer.merges
        tokenizer = super().converted(vocab, merges)

        tokenizer.add_special_tokens(
            [
                AddedToken("<|endoftext|>", normalized=False, special=True),
                AddedToken("<|im_start|>", normalized=False, special=True),
                AddedToken("<|im_end|>", normalized=False, special=True),
            ]
        )
        return tokenizer


class GGUFPhi3Converter(LlamaConverter):
    def __init__(self, tokenizer_dict):
        self.proto = GGUFTokenizerSkeleton(tokenizer_dict)
        self.original_tokenizer = self.proto
        self.additional_kwargs = {}

    def vocab(self, proto):
        return list(zip(proto.tokens, proto.scores))

    def merges(self, proto):
        return proto.merges

    def tokenizer(self, proto):
        vocab_scores = self.vocab(self.proto)
        merges = self.merges(self.proto)
        bpe_vocab = {word: i for i, (word, _score) in enumerate(vocab_scores)}

        tokenizer = Tokenizer(BPE(bpe_vocab, merges))
        # add the special tokens from phi3 tokenizer config
        tokenizer.add_special_tokens(
            [
                AddedToken("</s>", rstrip=True, lstrip=False, normalized=False, special=True),
                AddedToken("<|endoftext|>", normalized=False, special=True),
                AddedToken("<|assistant|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|placeholder1|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|placeholder2|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|placeholder3|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|placeholder4|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|system|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|end|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|placeholder5|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|placeholder6|>", rstrip=True, normalized=False, special=True),
                AddedToken("<|user|>", rstrip=True, normalized=False, special=True),
            ]
        )

        self.additional_kwargs["unk_token"] = (
            proto.tokens[proto.unk_token_id] if proto.unk_token_id is not None else None
        )
        self.additional_kwargs["eos_token"] = (
            proto.tokens[proto.eos_token_id] if proto.eos_token_id is not None else None
        )
        self.additional_kwargs["bos_token"] = (
            proto.tokens[proto.bos_token_id] if proto.bos_token_id is not None else None
        )
        self.additional_kwargs["pad_token"] = (
            proto.tokens[proto.pad_token_id] if proto.pad_token_id is not None else None
        )

        return tokenizer

    def decoder(self, replacement, add_prefix_space):
        sequence = [
            decoders.ByteFallback(),
            decoders.Fuse(),
            decoders.Replace(replacement, " "),
        ]

        if add_prefix_space:
            sequence += [decoders.Strip(content=" ", left=1)]
        return decoders.Sequence(sequence)

    def converted(self) -> Tokenizer:
        tokenizer = self.tokenizer(self.proto)

        replacement = "▁"
        add_prefix_space = True
        if hasattr(self.original_tokenizer, "add_prefix_space"):
            add_prefix_space = self.original_tokenizer.add_prefix_space

        tokenizer.decoder = self.decoder(replacement, add_prefix_space)

        return tokenizer


class GGUFGPTConverter(GPT2Converter):
    def __init__(self, tokenizer_dict):
        self.original_tokenizer = GGUFTokenizerSkeleton(tokenizer_dict)
        self.additional_kwargs = {}

    def converted(self) -> Tokenizer:
        vocab = {word: i for i, word in enumerate(self.original_tokenizer.tokens)}
        merges = self.original_tokenizer.merges
        tokenizer = super().converted(vocab, merges)
        return tokenizer


class GGUFT5Converter(T5Converter):
    def __init__(self, tokenizer_dict):
        # set dummy data to avoid unnecessary merges calculation
        tokenizer_dict["merges"] = ["dummy text"]

        self.proto = GGUFTokenizerSkeleton(tokenizer_dict)
        self.token2id = {k: v for v, k in enumerate(self.proto.tokens)}
        self.original_tokenizer = self.proto
        self.additional_kwargs = {}

    def vocab(self, proto):
        return list(zip(proto.tokens, proto.scores))

    def normalizer(self, proto):
        if getattr(self.original_tokenizer, "legacy", True):
            sequence = []
            if getattr(self.original_tokenizer, "add_prefix_space", True):
                sequence += [normalizers.Prepend(prepend="▁")]
            sequence += [normalizers.Replace(pattern=" ", content="▁")]
            return normalizers.Sequence(sequence)
        return None  # non-legacy, no normalizer

    def post_processor(self):
        return processors.TemplateProcessing(
            single=["$A", "</s>"],
            pair=["$A", "</s>", "$B", "</s>"],
            special_tokens=[
                ("</s>", self.token2id["</s>"]),
            ],
        )

    def converted(self) -> Tokenizer:
        vocab_scores = self.vocab(self.proto)
        tokenizer = Tokenizer(
            Unigram(
                vocab_scores,
                unk_id=self.proto.unk_token_id,
                byte_fallback=False,
            )
        )

        # Tokenizer assemble
        normalizer = self.normalizer(self.proto)
        if normalizer is not None:
            tokenizer.normalizer = normalizer

        replacement = "▁"
        add_prefix_space = True
        if hasattr(self.original_tokenizer, "add_prefix_space"):
            add_prefix_space = self.original_tokenizer.add_prefix_space

        pre_tokenizer = self.pre_tokenizer(replacement, add_prefix_space)
        if pre_tokenizer is not None:
            tokenizer.pre_tokenizer = pre_tokenizer

        tokenizer.decoder = self.decoder(replacement, add_prefix_space)
        post_processor = self.post_processor()
        if post_processor:
            tokenizer.post_processor = post_processor

        return tokenizer


GGUF_TO_FAST_CONVERTERS = {
    "llama": GGUFLlamaConverter,
    "qwen2": GGUFQwen2Converter,
    "qwen2_moe": GGUFQwen2Converter,
    "phi3": GGUFPhi3Converter,
    "bloom": GGUFGPTConverter,
    "falcon": GGUFGPTConverter,
    "stablelm": GGUFGPTConverter,
    "gpt2": GGUFGPTConverter,
    "starcoder2": GGUFGPTConverter,
    "t5": GGUFT5Converter,
    "mamba": GGUFGPTConverter,
    "nemotron": GGUFGPTConverter,
}


def convert_gguf_tokenizer(architecture, tokenizer_dict) -> Tokenizer:
    """
    Utilities to convert a slow tokenizer instance in a fast tokenizer instance.

    Args:
        architecture (`str`): The model architecture derived from gguf file.
        transformer_tokenizer ([`~tokenization_utils_base.PreTrainedTokenizer`]):
            Instance of a slow tokenizer to convert in the backend tokenizer for
            [`~tokenization_utils_base.PreTrainedTokenizerFast`].

    Return:
        A instance of [`~tokenizers.Tokenizer`] to be used as the backend tokenizer of a
        [`~tokenization_utils_base.PreTrainedTokenizerFast`]
    """
    tokenizer_class_name = architecture
    converter = GGUF_TO_FAST_CONVERTERS[tokenizer_class_name](tokenizer_dict)
    fast_tokenizer = converter.converted()
    return fast_tokenizer, converter.additional_kwargs
