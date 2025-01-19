# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
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
"""Convert Wav2Vec2Conformer checkpoint."""

import argparse
import json
import os

import fairseq
import torch
from fairseq.data import Dictionary

from transformers import (
    Wav2Vec2ConformerConfig,
    Wav2Vec2ConformerForCTC,
    Wav2Vec2ConformerForPreTraining,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    logging,
)


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

MAPPING = {
    "post_extract_proj": "feature_projection.projection",
    "encoder.pos_conv.0": "encoder.pos_conv_embed.conv",
    "self_attn.linear_k": "encoder.layers.*.self_attn.linear_k",
    "self_attn.linear_v": "encoder.layers.*.self_attn.linear_v",
    "self_attn.linear_q": "encoder.layers.*.self_attn.linear_q",
    "self_attn.pos_bias_u": "encoder.layers.*.self_attn.pos_bias_u",
    "self_attn.pos_bias_v": "encoder.layers.*.self_attn.pos_bias_v",
    "self_attn.linear_out": "encoder.layers.*.self_attn.linear_out",
    "self_attn.linear_pos": "encoder.layers.*.self_attn.linear_pos",
    "self_attn.rotary_emb": "encoder.embed_positions",
    "self_attn_layer_norm": "encoder.layers.*.self_attn_layer_norm",
    "conv_module.pointwise_conv1": "encoder.layers.*.conv_module.pointwise_conv1",
    "conv_module.pointwise_conv2": "encoder.layers.*.conv_module.pointwise_conv2",
    "conv_module.depthwise_conv": "encoder.layers.*.conv_module.depthwise_conv",
    "conv_module.batch_norm": "encoder.layers.*.conv_module.batch_norm",
    "conv_module.layer_norm": "encoder.layers.*.conv_module.layer_norm",
    "ffn1.w_1": "encoder.layers.*.ffn1.intermediate_dense",
    "ffn1.w_2": "encoder.layers.*.ffn1.output_dense",
    "ffn1.layer_norm": "encoder.layers.*.ffn1_layer_norm",
    "ffn2.w_1": "encoder.layers.*.ffn2.intermediate_dense",
    "ffn2.w_2": "encoder.layers.*.ffn2.output_dense",
    "ffn2.layer_norm": "encoder.layers.*.ffn2_layer_norm",
    "final_layer_norm": "encoder.layers.*.final_layer_norm",
    "encoder.layer_norm": "encoder.layer_norm",
    "w2v_model.layer_norm": "feature_projection.layer_norm",
    "quantizer.weight_proj": "quantizer.weight_proj",
    "quantizer.vars": "quantizer.codevectors",
    "project_q": "project_q",
    "final_proj": "project_hid",
    "w2v_encoder.proj": "lm_head",
    "mask_emb": "masked_spec_embed",
}
TOP_LEVEL_KEYS = [
    "lm_head",
    "quantizer.weight_proj",
    "quantizer.codevectors",
    "project_q",
    "project_hid",
]


def set_recursively(hf_pointer, key, value, full_name, weight_type):
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        hf_shape = hf_pointer.shape

    if hf_shape != value.shape:
        raise ValueError(
            f"Shape of hf {key + '.' + weight_type if weight_type is not None else ''} is {hf_shape}, but should be"
            f" {value.shape} for {full_name}"
        )

    if weight_type == "weight":
        hf_pointer.weight.data = value
    elif weight_type == "weight_g":
        hf_pointer.weight_g.data = value
    elif weight_type == "weight_v":
        hf_pointer.weight_v.data = value
    elif weight_type == "bias":
        hf_pointer.bias.data = value
    elif weight_type == "running_mean":
        hf_pointer.running_mean.data = value
    elif weight_type == "running_var":
        hf_pointer.running_var.data = value
    elif weight_type == "num_batches_tracked":
        hf_pointer.num_batches_tracked.data = value
    elif weight_type == "inv_freq":
        hf_pointer.inv_freq.data = value
    else:
        hf_pointer.data = value

    logger.info(f"{key + '.' + weight_type if weight_type is not None else ''} was initialized from {full_name}.")


def recursively_load_weights(fairseq_model, hf_model, is_headless):
    unused_weights = []
    fairseq_dict = fairseq_model.state_dict()

    feature_extractor = hf_model.wav2vec2_conformer.feature_extractor

    for name, value in fairseq_dict.items():
        is_used = False
        if "conv_layers" in name:
            load_conv_layer(
                name,
                value,
                feature_extractor,
                unused_weights,
                hf_model.config.feat_extract_norm == "group",
            )
            is_used = True
        else:
            for key, mapped_key in MAPPING.items():
                mapped_key = "wav2vec2_conformer." + mapped_key if mapped_key not in TOP_LEVEL_KEYS else mapped_key
                if key in name or key.split("w2v_model.")[-1] == name.split(".")[0]:
                    is_used = True
                    if "*" in mapped_key:
                        layer_index = name.split(key)[0].split(".")[-2]
                        mapped_key = mapped_key.replace("*", layer_index)
                    if "pos_bias_u" in name:
                        weight_type = None
                    elif "pos_bias_v" in name:
                        weight_type = None
                    elif "weight_g" in name:
                        weight_type = "weight_g"
                    elif "weight_v" in name:
                        weight_type = "weight_v"
                    elif "bias" in name:
                        weight_type = "bias"
                    elif "weight" in name:
                        # TODO: don't match quantizer.weight_proj
                        weight_type = "weight"
                    elif "running_mean" in name:
                        weight_type = "running_mean"
                    elif "inv_freq" in name:
                        weight_type = "inv_freq"
                    elif "running_var" in name:
                        weight_type = "running_var"
                    elif "num_batches_tracked" in name:
                        weight_type = "num_batches_tracked"
                    else:
                        weight_type = None
                    set_recursively(hf_model, mapped_key, value, name, weight_type)
                continue
        if not is_used:
            unused_weights.append(name)

    logger.warning(f"Unused weights: {unused_weights}")


# Copied from transformers.models.wav2vec2.convert_wav2vec2_original_pytorch_checkpoint_to_pytorch.load_conv_layer
def load_conv_layer(full_name, value, feature_extractor, unused_weights, use_group_norm):
    name = full_name.split("conv_layers.")[-1]
    items = name.split(".")
    layer_id = int(items[0])
    type_id = int(items[1])

    if type_id == 0:
        if "bias" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].conv.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.bias.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].conv.bias.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].conv.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].conv.weight.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].conv.weight.data = value
            logger.info(f"Feat extract conv layer {layer_id} was initialized from {full_name}.")
    elif (type_id == 2 and not use_group_norm) or (type_id == 2 and layer_id == 0 and use_group_norm):
        if "bias" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.bias.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].layer_norm.bias.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
        elif "weight" in name:
            if value.shape != feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape:
                raise ValueError(
                    f"{full_name} has size {value.shape}, but"
                    f" {feature_extractor.conv_layers[layer_id].layer_norm.weight.data.shape} was found."
                )
            feature_extractor.conv_layers[layer_id].layer_norm.weight.data = value
            logger.info(f"Feat extract layer norm weight of layer {layer_id} was initialized from {full_name}.")
    else:
        unused_weights.append(full_name)


@torch.no_grad()
def convert_wav2vec2_conformer_checkpoint(
    checkpoint_path,
    pytorch_dump_folder_path,
    sample_rate=16000,
    config_path=None,
    dict_path=None,
    is_finetuned=True,
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """

    if config_path is not None:
        config = Wav2Vec2ConformerConfig.from_pretrained(config_path, hidden_act="swish")
    else:
        config = Wav2Vec2ConformerConfig()

    if is_finetuned:
        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [checkpoint_path], arg_overrides={"data": "/".join(dict_path.split("/")[:-1])}
        )
    else:
        task_arg = argparse.Namespace(task="audio_pretraining")
        task = fairseq.tasks.setup_task(task_arg)

        model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path], task=task)

    model = model[0].eval()

    config.post_layer_norm = model.w2v_encoder.w2v_model.encoder.layer_norm_first

    if "rope" in checkpoint_path:
        config.position_embeddings_type = "rotary"

    if is_finetuned:
        if dict_path:
            target_dict = Dictionary.load(dict_path)

            # important change bos & pad token id since CTC symbol is <pad> and
            # not <s> as in fairseq
            config.bos_token_id = target_dict.pad_index
            config.pad_token_id = target_dict.bos_index
            config.eos_token_id = target_dict.eos_index
            config.vocab_size = len(target_dict.symbols)
            vocab_path = os.path.join(pytorch_dump_folder_path, "vocab.json")
            if not os.path.isdir(pytorch_dump_folder_path):
                logger.error("--pytorch_dump_folder_path ({}) should be a directory".format(pytorch_dump_folder_path))
                return
            os.makedirs(pytorch_dump_folder_path, exist_ok=True)
            vocab_dict = target_dict.indices

            # fairseq has the <pad> and <s> switched
            vocab_dict["<pad>"] = 0
            vocab_dict["<s>"] = 1
            with open(vocab_path, "w", encoding="utf-8") as vocab_handle:
                json.dump(vocab_dict, vocab_handle)
            tokenizer = Wav2Vec2CTCTokenizer(
                vocab_path,
                unk_token=target_dict.unk_word,
                pad_token=target_dict.pad_word,
                bos_token=target_dict.bos_word,
                eos_token=target_dict.eos_word,
                word_delimiter_token="|",
                do_lower_case=False,
            )
            return_attention_mask = True if config.feat_extract_norm == "layer" else False
            feature_extractor = Wav2Vec2FeatureExtractor(
                feature_size=1,
                sampling_rate=sample_rate,
                padding_value=0,
                do_normalize=True,
                return_attention_mask=return_attention_mask,
            )
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            processor.save_pretrained(pytorch_dump_folder_path)

        hf_wav2vec = Wav2Vec2ConformerForCTC(config)
    else:
        hf_wav2vec = Wav2Vec2ConformerForPreTraining(config)

    recursively_load_weights(model, hf_wav2vec, not is_finetuned)

    hf_wav2vec.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Path to fairseq checkpoint")
    parser.add_argument("--sample_rate", default=16000, type=int, help="Sample rate of the model")
    parser.add_argument("--dict_path", default=None, type=str, help="Path to dict of fine-tuned model")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument(
        "--not_finetuned", action="store_true", help="Whether the model to convert is a fine-tuned model or not"
    )
    args = parser.parse_args()
    convert_wav2vec2_conformer_checkpoint(
        args.checkpoint_path,
        args.pytorch_dump_folder_path,
        args.sample_rate,
        args.config_path,
        args.dict_path,
        not args.not_finetuned,
    )
