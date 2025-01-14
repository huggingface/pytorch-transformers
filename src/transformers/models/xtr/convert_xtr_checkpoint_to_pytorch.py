# coding=utf-8
# Copyright 2022 Google LLC and HuggingFace Inc. team.
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
Convert T5X checkpoint to PyTorch

Steps:
- Install gsutil according to https://cloud.google.com/storage/docs/gsutil_install
- Get a T5X checkpoint at https://github.com/google-research/t5x/blob/main/docs/models.md#t5-11-checkpoints Example:
    `gsutil -m cp -r gs://t5-data/pretrained_models/t5x/t5_1_1_small $HOME/`
- Create or download a corresponding config for the downloaded model. E.g. for T5 v1.1 small, you can use
    https://huggingface.co/google/t5-v1_1-small/blob/main/config.json
- Convert:
    ```
    python3 convert_t5x_checkpoint_to_pytorch.py --t5x_checkpoint_path=$HOME/t5_1_1_small --config_file=config.json\
      --pytorch_dump_path=$HOME/t5_1_1_small_pt
    ```
"""

import argparse
import collections

import torch
from flax import traverse_util
from t5x import checkpoints

from transformers import T5Config, T5EncoderModel, T5ForConditionalGeneration
from transformers.utils import logging


logging.set_verbosity_info()


def t5x_attention_lookup(params, i, prefix, layer_name="attention"):
    """Returns the KOQV parameters of (self-)attention. Does not transpose."""
    k = params[f"{prefix}/layers_{i}/{layer_name}/key/kernel"]
    o = params[f"{prefix}/layers_{i}/{layer_name}/out/kernel"]
    q = params[f"{prefix}/layers_{i}/{layer_name}/query/kernel"]
    v = params[f"{prefix}/layers_{i}/{layer_name}/value/kernel"]
    return k, o, q, v


def t5x_mlp_lookup(params, i, prefix, split_mlp_wi=False):
    """Returns the MLP parameters of a layer. Does not transpose."""
    if split_mlp_wi:
        wi_0 = params[f"{prefix}/layers_{i}/mlp/wi_0/kernel"]
        wi_1 = params[f"{prefix}/layers_{i}/mlp/wi_1/kernel"]
        wi = (wi_0, wi_1)
    else:
        wi = params[f"{prefix}/layers_{i}/mlp/wi/kernel"]

    wo = params[f"{prefix}/layers_{i}/mlp/wo/kernel"]
    return wi, wo


def t5x_layer_norm_lookup(params, i, prefix, layer_name):
    """Returns the layer norm param of a layer."""
    return params[f"{prefix}/layers_{i}/{layer_name}/scale"]


def convert_t5x_to_pytorch(variables: dict, *, num_layers: int, num_decoder_layers: int, is_encoder_only: bool):
    """Converts the parameters from T5X-Flax to Transformers-PyTorch."""
    old = traverse_util.flatten_dict(variables["target"])
    old = {"/".join(k): v for k, v in old.items()}

    # v1.1 models have a gated GeLU with wi_0 and wi_1 instead of wi
    split_mlp_wi = "encoder/layers_0/mlp/wi_0/kernel" in old
    print("Split MLP:", split_mlp_wi)

    new = collections.OrderedDict()

    # Shared embeddings.
    new["shared.weight"] = old["token_embedder/embedding"]

    # Encoder.
    for i in range(num_layers):
        # Block i, layer 0 (Self Attention).
        layer_norm = t5x_layer_norm_lookup(old, i, "encoder", "pre_attention_layer_norm")
        k, o, q, v = t5x_attention_lookup(old, i, "encoder", "attention")
        new[f"encoder.block.{i}.layer.0.layer_norm.weight"] = layer_norm
        new[f"encoder.block.{i}.layer.0.SelfAttention.k.weight"] = k.T
        new[f"encoder.block.{i}.layer.0.SelfAttention.o.weight"] = o.T
        new[f"encoder.block.{i}.layer.0.SelfAttention.q.weight"] = q.T
        new[f"encoder.block.{i}.layer.0.SelfAttention.v.weight"] = v.T

        # Block i, layer 1 (MLP).
        layer_norm = t5x_layer_norm_lookup(old, i, "encoder", "pre_mlp_layer_norm")
        wi, wo = t5x_mlp_lookup(old, i, "encoder", split_mlp_wi)
        new[f"encoder.block.{i}.layer.1.layer_norm.weight"] = layer_norm
        if split_mlp_wi:
            new[f"encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight"] = wi[0].T
            new[f"encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight"] = wi[1].T
        else:
            new[f"encoder.block.{i}.layer.1.DenseReluDense.wi.weight"] = wi.T
        new[f"encoder.block.{i}.layer.1.DenseReluDense.wo.weight"] = wo.T

    new["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = old[
        "encoder/relpos_bias/rel_embedding"
    ].T
    new["encoder.final_layer_norm.weight"] = old["encoder/encoder_norm/scale"]

    if not is_encoder_only:
        # Decoder.
        for i in range(num_decoder_layers):
            # Block i, layer 0 (Self Attention).
            layer_norm = t5x_layer_norm_lookup(old, i, "decoder", "pre_self_attention_layer_norm")
            k, o, q, v = t5x_attention_lookup(old, i, "decoder", "self_attention")
            new[f"decoder.block.{i}.layer.0.layer_norm.weight"] = layer_norm
            new[f"decoder.block.{i}.layer.0.SelfAttention.k.weight"] = k.T
            new[f"decoder.block.{i}.layer.0.SelfAttention.o.weight"] = o.T
            new[f"decoder.block.{i}.layer.0.SelfAttention.q.weight"] = q.T
            new[f"decoder.block.{i}.layer.0.SelfAttention.v.weight"] = v.T

            # Block i, layer 1 (Cross Attention).
            layer_norm = t5x_layer_norm_lookup(old, i, "decoder", "pre_cross_attention_layer_norm")
            k, o, q, v = t5x_attention_lookup(old, i, "decoder", "encoder_decoder_attention")
            new[f"decoder.block.{i}.layer.1.layer_norm.weight"] = layer_norm
            new[f"decoder.block.{i}.layer.1.EncDecAttention.k.weight"] = k.T
            new[f"decoder.block.{i}.layer.1.EncDecAttention.o.weight"] = o.T
            new[f"decoder.block.{i}.layer.1.EncDecAttention.q.weight"] = q.T
            new[f"decoder.block.{i}.layer.1.EncDecAttention.v.weight"] = v.T

            # Block i, layer 2 (MLP).
            layer_norm = t5x_layer_norm_lookup(old, i, "decoder", "pre_mlp_layer_norm")
            wi, wo = t5x_mlp_lookup(old, i, "decoder", split_mlp_wi)
            new[f"decoder.block.{i}.layer.2.layer_norm.weight"] = layer_norm
            if split_mlp_wi:
                new[f"decoder.block.{i}.layer.2.DenseReluDense.wi_0.weight"] = wi[0].T
                new[f"decoder.block.{i}.layer.2.DenseReluDense.wi_1.weight"] = wi[1].T
            else:
                new[f"decoder.block.{i}.layer.2.DenseReluDense.wi.weight"] = wi.T
            new[f"decoder.block.{i}.layer.2.DenseReluDense.wo.weight"] = wo.T

        new["decoder.final_layer_norm.weight"] = old["decoder/decoder_norm/scale"]
        new["decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"] = old[
            "decoder/relpos_bias/rel_embedding"
        ].T

        # LM Head (only in v1.1 checkpoints, in v1.0 embeddings are used instead)
        if "decoder/logits_dense/kernel" in old:
            new["lm_head.weight"] = old["decoder/logits_dense/kernel"].T

    return new


def make_state_dict(converted_params, is_encoder_only: bool):
    """Prepares a state dict for the PyTorch model."""
    # Make a state dict with torch tensors.
    state_dict = collections.OrderedDict([(k, torch.from_numpy(v.copy())) for (k, v) in converted_params.items()])

    # Add what is missing.
    if "encoder.embed_tokens.weight" not in state_dict:
        state_dict["encoder.embed_tokens.weight"] = state_dict["shared.weight"]

    if not is_encoder_only:
        if "decoder.embed_tokens.weight" not in state_dict:
            state_dict["decoder.embed_tokens.weight"] = state_dict["shared.weight"]

        if "lm_head.weight" not in state_dict:  # For old 1.0 models.
            print("Using shared word embeddings as lm_head.")
            state_dict["lm_head.weight"] = state_dict["shared.weight"]

    return state_dict


def load_t5x_weights_in_t5(model, config, t5x_checkpoint_path, is_encoder_only):
    """Replaces the params in model witht the T5X converted params."""
    variables = checkpoints.load_t5x_checkpoint(t5x_checkpoint_path)
    converted = convert_t5x_to_pytorch(
        variables,
        num_layers=config.num_layers,
        num_decoder_layers=config.num_decoder_layers,
        is_encoder_only=is_encoder_only,
    )
    state_dict = make_state_dict(converted, is_encoder_only)
    model.load_state_dict(state_dict, strict=True)


def convert_t5x_checkpoint_to_pytorch(
    t5x_checkpoint_path, config_file, pytorch_dump_path, is_encoder_only: bool = False
):
    """Loads the config and model, converts the T5X checkpoint, and saves a PyTorch checkpoint."""
    # Initialise PyTorch model
    config = T5Config.from_json_file(config_file)
    print(f"Building PyTorch model from configuration: {config}")
    # Non-v1.1 checkpoints could also use T5Model, but this works for all.
    # The v1.0 checkpoints will simply have an LM head that is the word embeddings.
    if is_encoder_only:
        model = T5EncoderModel(config)
    else:
        model = T5ForConditionalGeneration(config)

    # Load weights from tf checkpoint
    load_t5x_weights_in_t5(model, config, t5x_checkpoint_path, is_encoder_only)

    # Save pytorch-model
    print(f"Save PyTorch model to {pytorch_dump_path}")
    model.save_pretrained(pytorch_dump_path)

    # Verify that we can load the checkpoint.
    model.from_pretrained(pytorch_dump_path)
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts a native T5X checkpoint into a PyTorch checkpoint.")
    # Required parameters
    parser.add_argument(
        "--t5x_checkpoint_path", default=None, type=str, required=True, help="Path to the T5X checkpoint."
    )
    parser.add_argument(
        "--config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained T5 model.\nThis specifies the model architecture.",
    )
    parser.add_argument(
        "--pytorch_dump_path", default=None, type=str, required=True, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--is_encoder_only", action="store_true", help="Check if the model is encoder-decoder model", default=False
    )
    args = parser.parse_args()
    convert_t5x_checkpoint_to_pytorch(
        args.t5x_checkpoint_path, args.config_file, args.pytorch_dump_path, args.is_encoder_only
    )
