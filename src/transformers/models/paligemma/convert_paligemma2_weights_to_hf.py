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
"""Convert PaliGemma2 checkpoints from the original repository."""

import argparse
import collections

import jax.numpy as jnp
import ml_dtypes
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    Gemma2Config,
    PaliGemmaConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
    SiglipImageProcessor,
)
from transformers.tokenization_utils_base import AddedToken
from transformers.utils import logging


device = "cpu"

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

# TODO add sequence length variations here

PALIGEMMA2_VARIANTS = ["2b-224", "2b-448", "2b-896", "9b-224", "9b-448", "9b-896", "27b-224", "27b-448", "27b-896"]
VARIANT_CONFIGS = {
    "2b": {
        "num_positions": 256,
        "hidden_size": 2304,
        "num_hidden_layers": 26,
        "intermediate_size": 9216,
        "num_key_value_heads": 4,
        "num_attention_heads": 8,
        "head_dim": 256,
        "query_pre_attn_scalar": 256,
    },
    "9b": {
        "num_positions": 1024,
        "hidden_size": 3584,
        "num_hidden_layers": 42,
        "intermediate_size": 14336,
        "num_key_value_heads": 8,
        "num_attention_heads": 16,
        "head_dim": 256,
        "query_pre_attn_scalar": 256,
    },
    "27b": {
        "num_positions": 4096,
        "hidden_size": 4608,
        "num_hidden_layers": 46,
        "intermediate_size": 36864,
        "num_key_value_heads": 16,
        "num_attention_heads": 32,
        "head_dim": 128,
        "query_pre_attn_scalar": 4608 // 32,  # scaling is different for the 28b
    },
}

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def get_paligemma2_config(variant: str, precision: str):
    config = {
        "image_token_index": None,
        "pad_token_id": 0,
        "bos_token_id": 2,
        "eos_token_id": 1,
    }
    base_variant = variant.split("-")[0]

    if variant in PALIGEMMA2_VARIANTS:
        image_size = int(variant.split("-")[1])
        variant_config = VARIANT_CONFIGS[base_variant]
        patch_size = 14
        num_image_tokens = (image_size**2) // (patch_size**2)
        config["projection_dim"] = variant_config["hidden_size"]
        config["image_token_index"] = 257152
        config["num_hidden_layers"] = variant_config["num_hidden_layers"]  # For generate
        text_config = Gemma2Config.from_pretrained("google/gemma-2-2b-it").to_dict()
        sup_text_config = {
            "model_type": "gemma2",
            "vocab_size": 257152,
            "num_hidden_layers": variant_config["num_hidden_layers"],
            "num_key_value_heads": variant_config["num_key_value_heads"],
            "head_dim": variant_config["head_dim"],
            "torch_dtype": precision,
            "hidden_size": variant_config["hidden_size"],
            "hidden_activation": "gelu_pytorch_tanh",
            "num_attention_heads": variant_config["num_attention_heads"],
            "intermediate_size": variant_config["intermediate_size"],
            "is_encoder_decoder": False,
            "query_pre_attn_scalar": variant_config["query_pre_attn_scalar"],
        }
        text_config.update(sup_text_config)

        vision_config = {
            "num_positions": variant_config["num_positions"],  # not useful, to remove
            "torch_dtype": precision,
            "image_size": image_size,
            "patch_size": patch_size,
            "num_image_tokens": num_image_tokens,
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "num_hidden_layers": 27,
            "num_attention_heads": 16,
            "projection_dim": variant_config["hidden_size"],
            "hidden_act": "gelu_pytorch_tanh",
            "vision_use_head": False,
        }
        final_config = PaliGemmaConfig(text_config=text_config, vision_config=vision_config, **config)
    else:
        raise ValueError(f"Identifier {variant} not supported. Available: {PALIGEMMA2_VARIANTS}")
    return final_config


def slice_state_dict(state_dict, config):
    # fmt: off
    # patch embeddings
    state_dict["vision_tower.vision_model.embeddings.patch_embedding.weight"] = state_dict.pop("img/embedding/kernel").transpose(
        3, 2, 0, 1
    )
    state_dict["vision_tower.vision_model.embeddings.patch_embedding.bias"] = state_dict.pop("img/embedding/bias")
    # positional embeddings
    state_dict["vision_tower.vision_model.embeddings.position_embedding.weight"] = state_dict.pop("img/pos_embedding").reshape(
        -1, config.vision_config.hidden_size
    )

    # extract vision layers to be sliced at index 0. There are 27 layers in the base model.
    encoderblock_layernorm0_scale = state_dict.pop("img/Transformer/encoderblock/LayerNorm_0/scale")
    encoderblock_layernorm0_bias = state_dict.pop("img/Transformer/encoderblock/LayerNorm_0/bias")
    encoderblock_layernorm1_scale = state_dict.pop("img/Transformer/encoderblock/LayerNorm_1/scale")
    encoderblock_layernorm1_bias = state_dict.pop("img/Transformer/encoderblock/LayerNorm_1/bias")

    encoderblock_mlp_dense0_kernel= state_dict.pop("img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel")
    encoderblock_mlp_dense0_bias= state_dict.pop("img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias")
    encoderblock_mlp_dense1_kernel= state_dict.pop("img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel")
    encoderblock_mlp_dense1_bias= state_dict.pop("img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias")

    encoderblock_attention_0_key_kernel = state_dict.pop("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel")
    encoderblock_attention_0_key_bias = state_dict.pop("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias")
    encoderblock_attention_0_value_kernel = state_dict.pop("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel")
    encoderblock_attention_0_value_bias = state_dict.pop("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias")
    encoderblock_attention_0_query_kernel = state_dict.pop("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel")
    encoderblock_attention_0_query_bias = state_dict.pop("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias")
    encoderblock_attention_0_out_kernel = state_dict.pop("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel")
    encoderblock_attention_0_out_bias = state_dict.pop("img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias")

    for i in range(config.vision_config.num_hidden_layers):
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight"] = encoderblock_layernorm0_scale[i].transpose()
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.layer_norm1.bias"] = encoderblock_layernorm0_bias[i]
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight"] = encoderblock_layernorm1_scale[i].transpose()
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.layer_norm2.bias"] = encoderblock_layernorm1_bias[i]

        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight"] = encoderblock_mlp_dense0_kernel[i].transpose()
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.bias"] = encoderblock_mlp_dense0_bias[i]
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.weight"] = encoderblock_mlp_dense1_kernel[i].transpose()
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.bias"] = encoderblock_mlp_dense1_bias[i]
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"] = encoderblock_attention_0_key_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"] = encoderblock_attention_0_key_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"] = encoderblock_attention_0_value_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"] = encoderblock_attention_0_value_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"] = encoderblock_attention_0_query_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"] = encoderblock_attention_0_query_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"] = encoderblock_attention_0_out_kernel[i].reshape(-1, config.vision_config.hidden_size).transpose()
        state_dict[f"vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"] = encoderblock_attention_0_out_bias[i].reshape(-1, config.vision_config.hidden_size).reshape(-1)

    state_dict["vision_tower.vision_model.post_layernorm.weight"] = state_dict.pop("img/Transformer/encoder_norm/scale").transpose()
    state_dict["vision_tower.vision_model.post_layernorm.bias"] = state_dict.pop("img/Transformer/encoder_norm/bias")

    # multimodal projector

    state_dict['multi_modal_projector.linear.weight'] = state_dict.pop("img/head/kernel").transpose()
    state_dict['multi_modal_projector.linear.bias'] = state_dict.pop("img/head/bias")

    # text decoder (gemma)

    embedding_vector = state_dict.pop("llm/embedder/input_embedding")
    state_dict["language_model.model.embed_tokens.weight"] = embedding_vector

    # pop the einsum attention + mlp representations. There are 26 layers in gemma2-2b.

    llm_attention_attn_vec_einsum = state_dict.pop("llm/layers/attn/attn_vec_einsum/w")
    #  (26, 2, 4, 2304, 256) for 2b-224, 4 kv heads and 26 layers
    llm_attention_kv_einsum = state_dict.pop("llm/layers/attn/kv_einsum/w")
    llm_attention_q_einsum = state_dict.pop("llm/layers/attn/q_einsum/w")
    llm_mlp_gating_einsum = state_dict.pop("llm/layers/mlp/gating_einsum")
    llm_mlp_linear = state_dict.pop("llm/layers/mlp/linear")
    # TODO verify correctness of layer norm loading
    llm_input_layernorm = state_dict.pop("llm/layers/pre_attention_norm/scale")
    llm_pre_feedforward_layernorm = state_dict.pop("llm/layers/pre_ffw_norm/scale")

    llm_post_attention_layernorm = state_dict.pop("llm/layers/post_attention_norm/scale")
    llm_post_feedforward_layernorm = state_dict.pop("llm/layers/post_ffw_norm/scale")

    for i in range(config.text_config.num_hidden_layers):
        # llm_attention_q_einsum[i].shape = (8, 2048, 256)
        # q_proj_weight_reshaped = llm_attention_q_einsum[i].transpose(0, 2, 1).reshape(config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size)

        """
        q shape (8, 2304, 256)
        k shape (4, 2304, 256)
        v shape (4, 2304, 256)
        o shape (8, 256, 2304)

        """
        q_transpose = (0, 2, 1)
        k_transpose = (0, 2, 1)
        v_transpose = (0, 2, 1)
        o_transpose = (2, 0, 1)

        q_weight_matrices = llm_attention_q_einsum[i].transpose(*q_transpose)
        q_proj_weight_reshaped = q_weight_matrices
        q_proj_weight_reshaped = q_proj_weight_reshaped.reshape(config.text_config.num_attention_heads * config.text_config.head_dim, config.text_config.hidden_size)
        state_dict[f"language_model.model.layers.{i}.self_attn.q_proj.weight"] = q_proj_weight_reshaped
        # Shape: (4, 2304, 256)
        k_weight_matrices = llm_attention_kv_einsum[i, 0].transpose(*k_transpose)
        k_proj_weight_reshaped = k_weight_matrices.reshape(
            config.text_config.num_key_value_heads * config.text_config.head_dim,
            config.text_config.hidden_size
        )
        state_dict[f"language_model.model.layers.{i}.self_attn.k_proj.weight"] = k_proj_weight_reshaped
        # llm_attention_kv_einsum[i, 1].shape = (num_key_value_heads, hidden_size, head_dim)
        v_weight_matrices = llm_attention_kv_einsum[i, 1].transpose(*v_transpose) # Shape: (4, 2304, 256)
        v_proj_weight_reshaped = v_weight_matrices.reshape(
            config.text_config.num_key_value_heads * config.text_config.head_dim,
            config.text_config.hidden_size
        )
        state_dict[f"language_model.model.layers.{i}.self_attn.v_proj.weight"] = v_proj_weight_reshaped

        # output projection.

        # llm_attention_attn_vec_einsum[i].shape = (8, 256, 2304)
        o_proj_weight_reshaped = llm_attention_attn_vec_einsum[i].transpose(*o_transpose).reshape(config.text_config.hidden_size, config.text_config.num_attention_heads * config.text_config.head_dim)
        state_dict[f"language_model.model.layers.{i}.self_attn.o_proj.weight"] = o_proj_weight_reshaped
        # mlp layers
        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[f"language_model.model.layers.{i}.mlp.gate_proj.weight"] = gate_proj_weight.transpose()
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[f"language_model.model.layers.{i}.mlp.up_proj.weight"] = up_proj_weight.transpose()
        state_dict[f"language_model.model.layers.{i}.mlp.down_proj.weight"] = llm_mlp_linear[i].transpose()
        state_dict[f"language_model.model.layers.{i}.input_layernorm.weight"] = llm_input_layernorm[i]
        state_dict[f"language_model.model.layers.{i}.post_attention_layernorm.weight"] = llm_post_attention_layernorm[i]
        state_dict[f"language_model.model.layers.{i}.pre_feedforward_layernorm.weight"] = llm_pre_feedforward_layernorm[i]
        state_dict[f"language_model.model.layers.{i}.post_feedforward_layernorm.weight"] = llm_post_feedforward_layernorm[i]
    state_dict["language_model.model.norm.weight"] = state_dict.pop("llm/final_norm/scale")
    state_dict["language_model.lm_head.weight"] = embedding_vector # weights are tied.
    [k for k in state_dict.keys() if not k.startswith('vision') and not k.startswith('language')]
    # fmt: on
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            try:
                if value.dtype == jnp.bfloat16:
                    value = jnp.array(value).astype(jnp.float32)
                    value = np.array(value)
                    state_dict[key] = torch.from_numpy(value).to(torch.bfloat16)
                else:
                    state_dict[key] = torch.from_numpy(value)
            except Exception as initial_exception:
                raise ValueError(f"Conversion failed from jax weights with {initial_exception}. Check your inputs.")
    return state_dict


def flatten_nested_dict(params, parent_key="", sep="/", precision: int = "float32"):
    items = []

    for k, v in params.items():
        k = k.removeprefix("params/")
        new_key = parent_key + sep + k if parent_key else k

        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_nested_dict(v, parent_key=new_key, sep=sep, precision=precision).items())
        else:
            if precision == "bfloat16":
                try:
                    v = v.view(ml_dtypes.bfloat16)
                except Exception as initial_exception:
                    raise ValueError(f"Conversion failed from bfloat16 with {initial_exception}, check your inputs.")
            items.append((new_key, v))
    return dict(items)


@torch.no_grad()
def convert_paligemma2_checkpoint(
    checkpoint_path,
    pytorch_dump_folder_path,
    variant: str,
    precision: str,
    do_convert_weights=False,
):
    """
    Read checkpoints from flax npz files, rename/reshape, send result to state dict and verify logits if needed.
    """
    config = get_paligemma2_config(variant, precision=precision)
    if do_convert_weights:
        tokenizer_id = "google/paligemma-3b-pt-224"  # same tokenizer as paligemma 1
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        image_token = AddedToken("<image>", normalized=False, special=True)
        tokens_to_add = {"additional_special_tokens": [image_token]}
        tokenizer.add_special_tokens(tokens_to_add)

        # tokenizer.padding_side = 'right' # uncomment for testing purposes only.

        image_processor = SiglipImageProcessor.from_pretrained("google/paligemma-3b-pt-224")
        image_processor.size = {"width": config.vision_config.image_size, "height": config.vision_config.image_size}
        image_processor.image_seq_length = config.vision_config.num_image_tokens

        processor = PaliGemmaProcessor(image_processor=image_processor, tokenizer=tokenizer)
        data = jnp.load(checkpoint_path)
        state_dict = flatten_nested_dict(data, precision=precision)
        del data
        state_dict_transformers = slice_state_dict(state_dict, config)
        del state_dict
        del config.hidden_size  # this key is unused
        model = PaliGemmaForConditionalGeneration(config).to(device).eval()
        model.load_state_dict(state_dict_transformers)
        del state_dict_transformers
        model.config.text_config._attn_implementation = "sdpa"

        # model expansion to get random embeds of image tokens
        pad_shape = 64  # for performance reasons
        pre_expansion_embeddings = model.language_model.model.embed_tokens.weight.data
        mu = torch.mean(pre_expansion_embeddings, dim=0).float()
        n = pre_expansion_embeddings.size()[0]
        sigma = ((pre_expansion_embeddings - mu).T @ (pre_expansion_embeddings - mu)) / n
        dist = torch.distributions.multivariate_normal.MultivariateNormal(mu, covariance_matrix=1e-5 * sigma)

        # We add an image token so we resize the model
        model.resize_token_embeddings(config.text_config.vocab_size + 2, pad_shape)
        model.language_model.model.embed_tokens.weight.data[257152:] = torch.stack(
            tuple(
                (dist.sample() for _ in range(model.language_model.model.embed_tokens.weight.data[257152:].shape[0]))
            ),
            dim=0,
        )
        model.language_model.lm_head.weight.data[257152:] = torch.stack(
            tuple((dist.sample() for _ in range(model.language_model.lm_head.weight.data[257152:].shape[0]))),
            dim=0,
        )
        # convert to needed precision

        model.to(DTYPES[precision])
        model.save_pretrained(pytorch_dump_folder_path, safe_serialization=True)
        processor.save_pretrained(pytorch_dump_folder_path)

    else:
        processor = PaliGemmaProcessor.from_pretrained(pytorch_dump_folder_path, do_rescale=False)
        model = (
            PaliGemmaForConditionalGeneration.from_pretrained(pytorch_dump_folder_path, attn_implementation="sdpa")
            .to(device)
            .eval()
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to the .npz checkpoint",
    )

    parser.add_argument(
        "--pytorch_dump_folder_path",
        required=True,
        type=str,
        help="Path to the output directory where model and processor will be saved.",
    )

    parser.add_argument(
        "--precision",
        choices=["float32", "bfloat16", "float16"],
        type=str,
        help="Precision identifier for model conversion - should match the base checkpoint precision.",
    )

    parser.add_argument(
        "--variant",
        default="2b-224",
        choices=PALIGEMMA2_VARIANTS,
        type=str,
        help="String identifier of the paligemma2 variant to convert.",
    )

    parser.add_argument(
        "--do_convert_weights", action="store_true", help="Whether or not to reload and convert the weights."
    )

    args = parser.parse_args()
    convert_paligemma2_checkpoint(
        checkpoint_path=args.checkpoint_path,
        pytorch_dump_folder_path=args.pytorch_dump_folder_path,
        variant=args.variant,
        precision=args.precision,
        do_convert_weights=args.do_convert_weights,
    )
