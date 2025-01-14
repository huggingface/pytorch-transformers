# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Convert VITS checkpoint."""

import argparse
import json
import tempfile

import torch
from huggingface_hub import hf_hub_download

from transformers import VitsConfig, VitsModel, VitsTokenizer, logging


logging.set_verbosity_info()
logger = logging.get_logger("transformers.models.vits")

MAPPING_TEXT_ENCODER = {
    "enc_p.emb": "text_encoder.embed_tokens",
    "enc_p.encoder.attn_layers.*.conv_k": "text_encoder.encoder.layers.*.attention.k_proj",
    "enc_p.encoder.attn_layers.*.conv_v": "text_encoder.encoder.layers.*.attention.v_proj",
    "enc_p.encoder.attn_layers.*.conv_q": "text_encoder.encoder.layers.*.attention.q_proj",
    "enc_p.encoder.attn_layers.*.conv_o": "text_encoder.encoder.layers.*.attention.out_proj",
    "enc_p.encoder.attn_layers.*.emb_rel_k": "text_encoder.encoder.layers.*.attention.emb_rel_k",
    "enc_p.encoder.attn_layers.*.emb_rel_v": "text_encoder.encoder.layers.*.attention.emb_rel_v",
    "enc_p.encoder.norm_layers_1.*.gamma": "text_encoder.encoder.layers.*.layer_norm.weight",
    "enc_p.encoder.norm_layers_1.*.beta": "text_encoder.encoder.layers.*.layer_norm.bias",
    "enc_p.encoder.ffn_layers.*.conv_1": "text_encoder.encoder.layers.*.feed_forward.conv_1",
    "enc_p.encoder.ffn_layers.*.conv_2": "text_encoder.encoder.layers.*.feed_forward.conv_2",
    "enc_p.encoder.norm_layers_2.*.gamma": "text_encoder.encoder.layers.*.final_layer_norm.weight",
    "enc_p.encoder.norm_layers_2.*.beta": "text_encoder.encoder.layers.*.final_layer_norm.bias",
    "enc_p.proj": "text_encoder.project",
}
MAPPING_STOCHASTIC_DURATION_PREDICTOR = {
    "dp.pre": "duration_predictor.conv_pre",
    "dp.proj": "duration_predictor.conv_proj",
    "dp.convs.convs_sep.*": "duration_predictor.conv_dds.convs_dilated.*",
    "dp.convs.convs_1x1.*": "duration_predictor.conv_dds.convs_pointwise.*",
    "dp.convs.norms_1.*.gamma": "duration_predictor.conv_dds.norms_1.*.weight",
    "dp.convs.norms_1.*.beta": "duration_predictor.conv_dds.norms_1.*.bias",
    "dp.convs.norms_2.*.gamma": "duration_predictor.conv_dds.norms_2.*.weight",
    "dp.convs.norms_2.*.beta": "duration_predictor.conv_dds.norms_2.*.bias",
    "dp.flows.0.logs": "duration_predictor.flows.0.log_scale",
    "dp.flows.0.m": "duration_predictor.flows.0.translate",
    "dp.flows.*.pre": "duration_predictor.flows.*.conv_pre",
    "dp.flows.*.proj": "duration_predictor.flows.*.conv_proj",
    "dp.flows.*.convs.convs_1x1.0": "duration_predictor.flows.*.conv_dds.convs_pointwise.0",
    "dp.flows.*.convs.convs_1x1.1": "duration_predictor.flows.*.conv_dds.convs_pointwise.1",
    "dp.flows.*.convs.convs_1x1.2": "duration_predictor.flows.*.conv_dds.convs_pointwise.2",
    "dp.flows.*.convs.convs_sep.0": "duration_predictor.flows.*.conv_dds.convs_dilated.0",
    "dp.flows.*.convs.convs_sep.1": "duration_predictor.flows.*.conv_dds.convs_dilated.1",
    "dp.flows.*.convs.convs_sep.2": "duration_predictor.flows.*.conv_dds.convs_dilated.2",
    "dp.flows.*.convs.norms_1.0.gamma": "duration_predictor.flows.*.conv_dds.norms_1.0.weight",
    "dp.flows.*.convs.norms_1.0.beta": "duration_predictor.flows.*.conv_dds.norms_1.0.bias",
    "dp.flows.*.convs.norms_1.1.gamma": "duration_predictor.flows.*.conv_dds.norms_1.1.weight",
    "dp.flows.*.convs.norms_1.1.beta": "duration_predictor.flows.*.conv_dds.norms_1.1.bias",
    "dp.flows.*.convs.norms_1.2.gamma": "duration_predictor.flows.*.conv_dds.norms_1.2.weight",
    "dp.flows.*.convs.norms_1.2.beta": "duration_predictor.flows.*.conv_dds.norms_1.2.bias",
    "dp.flows.*.convs.norms_2.0.gamma": "duration_predictor.flows.*.conv_dds.norms_2.0.weight",
    "dp.flows.*.convs.norms_2.0.beta": "duration_predictor.flows.*.conv_dds.norms_2.0.bias",
    "dp.flows.*.convs.norms_2.1.gamma": "duration_predictor.flows.*.conv_dds.norms_2.1.weight",
    "dp.flows.*.convs.norms_2.1.beta": "duration_predictor.flows.*.conv_dds.norms_2.1.bias",
    "dp.flows.*.convs.norms_2.2.gamma": "duration_predictor.flows.*.conv_dds.norms_2.2.weight",
    "dp.flows.*.convs.norms_2.2.beta": "duration_predictor.flows.*.conv_dds.norms_2.2.bias",
    "dp.post_pre": "duration_predictor.post_conv_pre",
    "dp.post_proj": "duration_predictor.post_conv_proj",
    "dp.post_convs.convs_sep.*": "duration_predictor.post_conv_dds.convs_dilated.*",
    "dp.post_convs.convs_1x1.*": "duration_predictor.post_conv_dds.convs_pointwise.*",
    "dp.post_convs.norms_1.*.gamma": "duration_predictor.post_conv_dds.norms_1.*.weight",
    "dp.post_convs.norms_1.*.beta": "duration_predictor.post_conv_dds.norms_1.*.bias",
    "dp.post_convs.norms_2.*.gamma": "duration_predictor.post_conv_dds.norms_2.*.weight",
    "dp.post_convs.norms_2.*.beta": "duration_predictor.post_conv_dds.norms_2.*.bias",
    "dp.post_flows.0.logs": "duration_predictor.post_flows.0.log_scale",
    "dp.post_flows.0.m": "duration_predictor.post_flows.0.translate",
    "dp.post_flows.*.pre": "duration_predictor.post_flows.*.conv_pre",
    "dp.post_flows.*.proj": "duration_predictor.post_flows.*.conv_proj",
    "dp.post_flows.*.convs.convs_1x1.0": "duration_predictor.post_flows.*.conv_dds.convs_pointwise.0",
    "dp.post_flows.*.convs.convs_1x1.1": "duration_predictor.post_flows.*.conv_dds.convs_pointwise.1",
    "dp.post_flows.*.convs.convs_1x1.2": "duration_predictor.post_flows.*.conv_dds.convs_pointwise.2",
    "dp.post_flows.*.convs.convs_sep.0": "duration_predictor.post_flows.*.conv_dds.convs_dilated.0",
    "dp.post_flows.*.convs.convs_sep.1": "duration_predictor.post_flows.*.conv_dds.convs_dilated.1",
    "dp.post_flows.*.convs.convs_sep.2": "duration_predictor.post_flows.*.conv_dds.convs_dilated.2",
    "dp.post_flows.*.convs.norms_1.0.gamma": "duration_predictor.post_flows.*.conv_dds.norms_1.0.weight",
    "dp.post_flows.*.convs.norms_1.0.beta": "duration_predictor.post_flows.*.conv_dds.norms_1.0.bias",
    "dp.post_flows.*.convs.norms_1.1.gamma": "duration_predictor.post_flows.*.conv_dds.norms_1.1.weight",
    "dp.post_flows.*.convs.norms_1.1.beta": "duration_predictor.post_flows.*.conv_dds.norms_1.1.bias",
    "dp.post_flows.*.convs.norms_1.2.gamma": "duration_predictor.post_flows.*.conv_dds.norms_1.2.weight",
    "dp.post_flows.*.convs.norms_1.2.beta": "duration_predictor.post_flows.*.conv_dds.norms_1.2.bias",
    "dp.post_flows.*.convs.norms_2.0.gamma": "duration_predictor.post_flows.*.conv_dds.norms_2.0.weight",
    "dp.post_flows.*.convs.norms_2.0.beta": "duration_predictor.post_flows.*.conv_dds.norms_2.0.bias",
    "dp.post_flows.*.convs.norms_2.1.gamma": "duration_predictor.post_flows.*.conv_dds.norms_2.1.weight",
    "dp.post_flows.*.convs.norms_2.1.beta": "duration_predictor.post_flows.*.conv_dds.norms_2.1.bias",
    "dp.post_flows.*.convs.norms_2.2.gamma": "duration_predictor.post_flows.*.conv_dds.norms_2.2.weight",
    "dp.post_flows.*.convs.norms_2.2.beta": "duration_predictor.post_flows.*.conv_dds.norms_2.2.bias",
    "dp.cond": "duration_predictor.cond",  # num_speakers > 1
}
MAPPING_FLOW = {
    "flow.flows.*.pre": "flow.flows.*.conv_pre",
    "flow.flows.*.enc.in_layers.0": "flow.flows.*.wavenet.in_layers.0",
    "flow.flows.*.enc.in_layers.1": "flow.flows.*.wavenet.in_layers.1",
    "flow.flows.*.enc.in_layers.2": "flow.flows.*.wavenet.in_layers.2",
    "flow.flows.*.enc.in_layers.3": "flow.flows.*.wavenet.in_layers.3",
    "flow.flows.*.enc.res_skip_layers.0": "flow.flows.*.wavenet.res_skip_layers.0",
    "flow.flows.*.enc.res_skip_layers.1": "flow.flows.*.wavenet.res_skip_layers.1",
    "flow.flows.*.enc.res_skip_layers.2": "flow.flows.*.wavenet.res_skip_layers.2",
    "flow.flows.*.enc.res_skip_layers.3": "flow.flows.*.wavenet.res_skip_layers.3",
    "flow.flows.*.enc.cond_layer": "flow.flows.*.wavenet.cond_layer",  # num_speakers > 1
    "flow.flows.*.post": "flow.flows.*.conv_post",
}
MAPPING_GENERATOR = {
    "dec.conv_pre": "decoder.conv_pre",
    "dec.ups.0": "decoder.upsampler.0",
    "dec.ups.1": "decoder.upsampler.1",
    "dec.ups.2": "decoder.upsampler.2",
    "dec.ups.3": "decoder.upsampler.3",
    "dec.resblocks.*.convs1.0": "decoder.resblocks.*.convs1.0",
    "dec.resblocks.*.convs1.1": "decoder.resblocks.*.convs1.1",
    "dec.resblocks.*.convs1.2": "decoder.resblocks.*.convs1.2",
    "dec.resblocks.*.convs2.0": "decoder.resblocks.*.convs2.0",
    "dec.resblocks.*.convs2.1": "decoder.resblocks.*.convs2.1",
    "dec.resblocks.*.convs2.2": "decoder.resblocks.*.convs2.2",
    "dec.conv_post": "decoder.conv_post",
    "dec.multistream_conv_post": "decoder.multistream_conv_post",
    "dec.cond": "decoder.cond",  # num_speakers > 1
}
MAPPING_POSTERIOR_ENCODER = {
    "enc_q.pre": "posterior_encoder.conv_pre",
    "enc_q.enc.in_layers.*": "posterior_encoder.wavenet.in_layers.*",
    "enc_q.enc.res_skip_layers.*": "posterior_encoder.wavenet.res_skip_layers.*",
    "enc_q.enc.cond_layer": "posterior_encoder.wavenet.cond_layer",  # num_speakers > 1
    "enc_q.proj": "posterior_encoder.conv_proj",
}
MAPPING = {
    **MAPPING_TEXT_ENCODER,
    **MAPPING_STOCHASTIC_DURATION_PREDICTOR,
    **MAPPING_FLOW,
    **MAPPING_GENERATOR,
    **MAPPING_POSTERIOR_ENCODER,
    "emb_g": "embed_speaker",  # num_speakers > 1
}
TOP_LEVEL_KEYS = []
IGNORE_KEYS = []


def set_recursively(hf_pointer, key, value, full_name, weight_type):
    for attribute in key.split("."):
        hf_pointer = getattr(hf_pointer, attribute)

    if weight_type is not None:
        hf_shape = getattr(hf_pointer, weight_type).shape
    else:
        hf_shape = hf_pointer.shape

    # strip off the kernel dimension at the end (original weights are Conv1d)
    if key.endswith(".k_proj") or key.endswith(".v_proj") or key.endswith(".q_proj") or key.endswith(".out_proj"):
        value = value.squeeze(-1)

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
    else:
        hf_pointer.data = value

    logger.info(f"{key + ('.' + weight_type if weight_type is not None else '')} was initialized from {full_name}.")


def should_ignore(name, ignore_keys):
    for key in ignore_keys:
        if key.endswith(".*"):
            if name.startswith(key[:-1]):
                return True
        elif ".*." in key:
            prefix, suffix = key.split(".*.")
            if prefix in name and suffix in name:
                return True
        elif key in name:
            return True
    return False


def recursively_load_weights(fairseq_dict, hf_model):
    unused_weights = []

    for name, value in fairseq_dict.items():
        if should_ignore(name, IGNORE_KEYS):
            logger.info(f"{name} was ignored")
            continue

        is_used = False
        for key, mapped_key in MAPPING.items():
            if key.endswith(".*"):
                key = key[:-1]
            elif "*" in key:
                prefix, suffix = key.split(".*.")
                if prefix in name and suffix in name:
                    key = suffix

            if key in name:
                print(key, name)
                is_used = True
                if mapped_key.endswith(".*"):
                    layer_index = name.split(key)[-1].split(".")[0]
                    mapped_key = mapped_key.replace("*", layer_index)
                elif "*" in mapped_key:
                    layer_index = name.split(key)[0].split(".")[-2]

                    # remap the layer index since we removed the Flip layers
                    if "flow.flows" in mapped_key:
                        layer_index = str(int(layer_index) // 2)
                    if "duration_predictor.flows" in mapped_key or "duration_predictor.post_flows" in mapped_key:
                        layer_index = str(int(layer_index) // 2 + 1)

                    mapped_key = mapped_key.replace("*", layer_index)
                if "weight_g" in name:
                    weight_type = "weight_g"
                elif "weight_v" in name:
                    weight_type = "weight_v"
                elif "bias" in name:
                    weight_type = "bias"
                elif "weight" in name:
                    weight_type = "weight"
                elif "running_mean" in name:
                    weight_type = "running_mean"
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


@torch.no_grad()
def convert_checkpoint(
    pytorch_dump_folder_path,
    checkpoint_path=None,
    config_path=None,
    vocab_path=None,
    language=None,
    num_speakers=None,
    istft_decoder=None,
    sampling_rate=None,
    repo_id=None,
):
    """
    Copy/paste/tweak model's weights to transformers design.
    """
    if config_path is not None:
        config = VitsConfig.from_pretrained(config_path)
    else:
        config = VitsConfig()

    if num_speakers:
        config.num_speakers = num_speakers
        config.speaker_embedding_size = 256

    if sampling_rate:
        config.sampling_rate = sampling_rate

    if istft_decoder:
        if istft_decoder == "istft":
            config.istft_decoder = istft_decoder
            config.gen_istft_hop_size = 4
            config.gen_istft_n_fft = 16
            config.upsample_rates = [8, 8]
            config.upsample_kernel_sizes = [16, 16]
        elif istft_decoder in ["mb_istft", "ms_istft"]:
            config.istft_decoder = istft_decoder
            config.gen_istft_hop_size = 4
            config.gen_istft_n_fft = 16
            config.upsample_rates = [4, 4]
            config.upsample_kernel_sizes = [16, 16]
            config.subbands = 4

    if checkpoint_path is None:
        logger.info(f"***Converting model: facebook/mms-tts {language}***")

        vocab_path = hf_hub_download(
            repo_id="facebook/mms-tts",
            filename="vocab.txt",
            subfolder=f"models/{language}",
        )
        config_file = hf_hub_download(
            repo_id="facebook/mms-tts",
            filename="config.json",
            subfolder=f"models/{language}",
        )
        checkpoint_path = hf_hub_download(
            repo_id="facebook/mms-tts",
            filename="G_100000.pth",
            subfolder=f"models/{language}",
        )

        with open(config_file, "r") as f:
            data = f.read()
            hps = json.loads(data)

        is_uroman = hps["data"]["training_files"].split(".")[-1] == "uroman"
        if is_uroman:
            logger.warning("For this checkpoint, you should use `uroman` to convert input text before tokenizing it!")
    else:
        logger.info(f"***Converting model: {checkpoint_path}***")
        is_uroman = False

    # original VITS checkpoint
    if vocab_path is None:
        _pad = "_"
        _punctuation = ';:,.!?¡¿—…"«»“” '
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
        symbols = _pad + _punctuation + _letters + _letters_ipa
        symbol_to_id = {s: i for i, s in enumerate(symbols)}
        phonemize = True
    else:
        # Save vocab as temporary json file
        symbols = [line.replace("\n", "") for line in open(vocab_path, encoding="utf-8").readlines()]
        symbol_to_id = {s: i for i, s in enumerate(symbols)}
        # MMS-TTS does not use a <pad> token, so we set to the token used to space characters
        _pad = symbols[0]
        phonemize = False

    with tempfile.NamedTemporaryFile() as tf:
        with open(tf.name, "w", encoding="utf-8") as f:
            f.write(json.dumps(symbol_to_id, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

        tokenizer = VitsTokenizer(tf.name, language=language, phonemize=phonemize, is_uroman=is_uroman, pad_token=_pad)

    config.vocab_size = len(symbols)
    model = VitsModel(config)

    model.decoder.apply_weight_norm()

    orig_checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    if istft_decoder in ["mb_istft", "ms_istft"]:
        orig_checkpoint["model"]["dec.conv_post.bias"] = orig_checkpoint["model"]["dec.subband_conv_post.bias"]
        orig_checkpoint["model"]["dec.conv_post.weight_g"] = orig_checkpoint["model"]["dec.subband_conv_post.weight_g"]
        orig_checkpoint["model"]["dec.conv_post.weight_v"] = orig_checkpoint["model"]["dec.subband_conv_post.weight_v"]
        del orig_checkpoint["model"]["dec.subband_conv_post.bias"]
        del orig_checkpoint["model"]["dec.subband_conv_post.weight_g"]
        del orig_checkpoint["model"]["dec.subband_conv_post.weight_v"]
    recursively_load_weights(orig_checkpoint["model"], model)

    model.decoder.remove_weight_norm()

    model.save_pretrained(pytorch_dump_folder_path)
    tokenizer.save_pretrained(pytorch_dump_folder_path)

    if repo_id:
        print("Pushing to the hub...")
        tokenizer.push_to_hub(repo_id)
        model.push_to_hub(repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default=None, type=str, help="Local path to original checkpoint")
    parser.add_argument("--vocab_path", default=None, type=str, help="Path to vocab.txt")
    parser.add_argument("--config_path", default=None, type=str, help="Path to hf config.json of model to convert")
    parser.add_argument("--language", default=None, type=str, help="Tokenizer language (three-letter code)")
    parser.add_argument("--num_speakers", default=None, type=int, help="Number of speakers")
    parser.add_argument(
        "--istft_decoder",
        default=None,
        type=str,
        choices=["istft", "ms_istft", "mb_istft"],
        help="ISTFT decoder options for vits",
    )

    parser.add_argument(
        "--sampling_rate", default=None, type=int, help="Sampling rate on which the model was trained."
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", required=True, default=None, type=str, help="Path to the output PyTorch model."
    )
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Where to upload the converted model on the 🤗 hub."
    )

    args = parser.parse_args()
    convert_checkpoint(
        args.pytorch_dump_folder_path,
        args.checkpoint_path,
        args.config_path,
        args.vocab_path,
        args.language,
        args.num_speakers,
        args.istft_decoder,
        args.sampling_rate,
        args.push_to_hub,
    )
