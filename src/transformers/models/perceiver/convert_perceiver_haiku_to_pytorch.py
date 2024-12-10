# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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
"""Convert Perceiver checkpoints originally implemented in Haiku."""

import argparse
import json
import pickle
from pathlib import Path

import haiku as hk
import numpy as np
import requests
import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from transformers import (
    PerceiverConfig,
    PerceiverForImageClassificationConvProcessing,
    PerceiverForImageClassificationFourier,
    PerceiverForImageClassificationLearned,
    PerceiverForMaskedLM,
    PerceiverForMultimodalAutoencoding,
    PerceiverForOpticalFlow,
    PerceiverImageProcessor,
    PerceiverTokenizer,
)
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def prepare_img():
    # We will verify our results on an image of a dog
    url = "https://storage.googleapis.com/perceiver_io/dalmation.jpg"
    im = Image.open(requests.get(url, stream=True).raw)
    return im


def rename_keys(state_dict, architecture):
    for name in list(state_dict):
        param = state_dict.pop(name)

        # PREPROCESSORS
        # rename text preprocessor embeddings (for MLM model)
        name = name.replace("embed/embeddings", "input_preprocessor.embeddings.weight")
        if name.startswith("trainable_position_encoding/pos_embs"):
            name = name.replace(
                "trainable_position_encoding/pos_embs", "input_preprocessor.position_embeddings.weight"
            )

        # rename image preprocessor embeddings (for image classification model with learned position embeddings)
        name = name.replace("image_preprocessor/~/conv2_d/w", "input_preprocessor.convnet_1x1.weight")
        name = name.replace("image_preprocessor/~/conv2_d/b", "input_preprocessor.convnet_1x1.bias")
        name = name.replace(
            "image_preprocessor/~_build_network_inputs/trainable_position_encoding/pos_embs",
            "input_preprocessor.position_embeddings.position_embeddings",
        )
        name = name.replace(
            "image_preprocessor/~_build_network_inputs/position_encoding_projector/linear/w",
            "input_preprocessor.positions_projection.weight",
        )
        name = name.replace(
            "image_preprocessor/~_build_network_inputs/position_encoding_projector/linear/b",
            "input_preprocessor.positions_projection.bias",
        )

        # rename image preprocessor embeddings (for image classification model with conv processing)
        if "counter" in name or "hidden" in name:
            continue
        name = name.replace(
            "image_preprocessor/~/conv2_d_downsample/~/conv/w", "input_preprocessor.convnet.conv.weight"
        )
        name = name.replace(
            "image_preprocessor/~/conv2_d_downsample/~/batchnorm/offset", "input_preprocessor.convnet.batchnorm.bias"
        )
        name = name.replace(
            "image_preprocessor/~/conv2_d_downsample/~/batchnorm/scale", "input_preprocessor.convnet.batchnorm.weight"
        )
        name = name.replace(
            "image_preprocessor/~/conv2_d_downsample/~/batchnorm/~/mean_ema/average",
            "input_preprocessor.convnet.batchnorm.running_mean",
        )
        name = name.replace(
            "image_preprocessor/~/conv2_d_downsample/~/batchnorm/~/var_ema/average",
            "input_preprocessor.convnet.batchnorm.running_var",
        )

        # rename image preprocessor embeddings (for optical flow model)
        name = name.replace("image_preprocessor/patches_linear/b", "input_preprocessor.conv_after_patches.bias")
        name = name.replace("image_preprocessor/patches_linear/w", "input_preprocessor.conv_after_patches.weight")

        # rename multimodal preprocessor embeddings
        name = name.replace("multimodal_preprocessor/audio_mask_token/pos_embs", "input_preprocessor.mask.audio")
        name = name.replace("multimodal_preprocessor/audio_padding/pos_embs", "input_preprocessor.padding.audio")
        name = name.replace("multimodal_preprocessor/image_mask_token/pos_embs", "input_preprocessor.mask.image")
        name = name.replace("multimodal_preprocessor/image_padding/pos_embs", "input_preprocessor.padding.image")
        name = name.replace("multimodal_preprocessor/label_mask_token/pos_embs", "input_preprocessor.mask.label")
        name = name.replace("multimodal_preprocessor/label_padding/pos_embs", "input_preprocessor.padding.label")

        # DECODERS
        # rename prefix of decoders
        # multimodal autoencoding model
        name = name.replace(
            "multimodal_decoder/~/basic_decoder/cross_attention/", "decoder.decoder.decoding_cross_attention."
        )
        name = name.replace("multimodal_decoder/~decoder_query/audio_padding/pos_embs", "decoder.padding.audio")
        name = name.replace("multimodal_decoder/~decoder_query/image_padding/pos_embs", "decoder.padding.image")
        name = name.replace("multimodal_decoder/~decoder_query/label_padding/pos_embs", "decoder.padding.label")
        name = name.replace("multimodal_decoder/~/basic_decoder/output/b", "decoder.decoder.final_layer.bias")
        name = name.replace("multimodal_decoder/~/basic_decoder/output/w", "decoder.decoder.final_layer.weight")
        if architecture == "multimodal_autoencoding":
            name = name.replace(
                "classification_decoder/~/basic_decoder/~/trainable_position_encoding/pos_embs",
                "decoder.modalities.label.decoder.output_position_encodings.position_embeddings",
            )
        # flow model
        name = name.replace(
            "flow_decoder/~/basic_decoder/cross_attention/", "decoder.decoder.decoding_cross_attention."
        )
        name = name.replace("flow_decoder/~/basic_decoder/output/w", "decoder.decoder.final_layer.weight")
        name = name.replace("flow_decoder/~/basic_decoder/output/b", "decoder.decoder.final_layer.bias")
        # image models
        name = name.replace(
            "classification_decoder/~/basic_decoder/~/trainable_position_encoding/pos_embs",
            "decoder.decoder.output_position_encodings.position_embeddings",
        )
        name = name.replace(
            "basic_decoder/~/trainable_position_encoding/pos_embs",
            "decoder.output_position_encodings.position_embeddings",
        )
        name = name.replace(
            "classification_decoder/~/basic_decoder/cross_attention/", "decoder.decoder.decoding_cross_attention."
        )
        name = name.replace("classification_decoder/~/basic_decoder/output/b", "decoder.decoder.final_layer.bias")
        name = name.replace("classification_decoder/~/basic_decoder/output/w", "decoder.decoder.final_layer.weight")
        name = name = name.replace("classification_decoder/~/basic_decoder/~/", "decoder.decoder.")
        name = name.replace("basic_decoder/cross_attention/", "decoder.decoding_cross_attention.")
        name = name.replace("basic_decoder/~/", "decoder.")

        # POSTPROCESSORS
        name = name.replace(
            "projection_postprocessor/linear/b", "output_postprocessor.modalities.image.classifier.bias"
        )
        name = name.replace(
            "projection_postprocessor/linear/w", "output_postprocessor.modalities.image.classifier.weight"
        )
        name = name.replace(
            "classification_postprocessor/linear/b", "output_postprocessor.modalities.label.classifier.bias"
        )
        name = name.replace(
            "classification_postprocessor/linear/w", "output_postprocessor.modalities.label.classifier.weight"
        )
        name = name.replace("audio_postprocessor/linear/b", "output_postprocessor.modalities.audio.classifier.bias")
        name = name.replace("audio_postprocessor/linear/w", "output_postprocessor.modalities.audio.classifier.weight")

        # PERCEIVER MODEL

        # rename latent embeddings
        name = name.replace("perceiver_encoder/~/trainable_position_encoding/pos_embs", "embeddings.latents")
        # rename latent embeddings (for multimodal model)
        name = name.replace("encoder/~/trainable_position_encoding/pos_embs", "embeddings.latents")

        # rename prefixes
        if name.startswith("perceiver_encoder/~/"):
            if "self_attention" in name:
                suffix = "self_attends."
            else:
                suffix = ""
            name = name.replace("perceiver_encoder/~/", "encoder." + suffix)
        if name.startswith("encoder/~/"):
            if "self_attention" in name:
                suffix = "self_attends."
            else:
                suffix = ""
            name = name.replace("encoder/~/", "encoder." + suffix)
        # rename layernorm parameters
        if "offset" in name:
            name = name.replace("offset", "bias")
        if "scale" in name:
            name = name.replace("scale", "weight")
        # in HuggingFace, the layernorm in between attention + MLP is just called "layernorm"
        # rename layernorm in between attention + MLP of cross-attention
        if "cross_attention" in name and "layer_norm_2" in name:
            name = name.replace("layer_norm_2", "layernorm")
        # rename layernorm in between attention + MLP of self-attention
        if "self_attention" in name and "layer_norm_1" in name:
            name = name.replace("layer_norm_1", "layernorm")

        # in HuggingFace, the layernorms for queries + keys are called "layernorm1" and "layernorm2"
        if "cross_attention" in name and "layer_norm_1" in name:
            name = name.replace("layer_norm_1", "attention.self.layernorm2")
        if "cross_attention" in name and "layer_norm" in name:
            name = name.replace("layer_norm", "attention.self.layernorm1")
        if "self_attention" in name and "layer_norm" in name:
            name = name.replace("layer_norm", "attention.self.layernorm1")

        # rename special characters by dots
        name = name.replace("-", ".")
        name = name.replace("/", ".")
        # rename keys, queries, values and output of attention layers
        if ("cross_attention" in name or "self_attention" in name) and "mlp" not in name:
            if "linear.b" in name:
                name = name.replace("linear.b", "self.query.bias")
            if "linear.w" in name:
                name = name.replace("linear.w", "self.query.weight")
            if "linear_1.b" in name:
                name = name.replace("linear_1.b", "self.key.bias")
            if "linear_1.w" in name:
                name = name.replace("linear_1.w", "self.key.weight")
            if "linear_2.b" in name:
                name = name.replace("linear_2.b", "self.value.bias")
            if "linear_2.w" in name:
                name = name.replace("linear_2.w", "self.value.weight")
            if "linear_3.b" in name:
                name = name.replace("linear_3.b", "output.dense.bias")
            if "linear_3.w" in name:
                name = name.replace("linear_3.w", "output.dense.weight")
        if "self_attention_" in name:
            name = name.replace("self_attention_", "")
        if "self_attention" in name:
            name = name.replace("self_attention", "0")
        # rename dense layers of 2-layer MLP
        if "mlp" in name:
            if "linear.b" in name:
                name = name.replace("linear.b", "dense1.bias")
            if "linear.w" in name:
                name = name.replace("linear.w", "dense1.weight")
            if "linear_1.b" in name:
                name = name.replace("linear_1.b", "dense2.bias")
            if "linear_1.w" in name:
                name = name.replace("linear_1.w", "dense2.weight")

        # finally, TRANSPOSE if kernel and not embedding layer, and set value
        if name[-6:] == "weight" and "embeddings" not in name:
            param = np.transpose(param)

        # if batchnorm, we need to squeeze it
        if "batchnorm" in name:
            param = np.squeeze(param)

        if "embedding_decoder" not in name:
            state_dict["perceiver." + name] = torch.from_numpy(param)
        else:
            state_dict[name] = torch.from_numpy(param)


@torch.no_grad()
def convert_perceiver_checkpoint(pickle_file, pytorch_dump_folder_path, architecture="MLM"):
    """
    Copy/paste/tweak model's weights to our Perceiver structure.
    """

    # load parameters as FlatMapping data structure
    with open(pickle_file, "rb") as f:
        checkpoint = pickle.loads(f.read())

    state = None
    if isinstance(checkpoint, dict) and architecture in [
        "image_classification",
        "image_classification_fourier",
        "image_classification_conv",
    ]:
        # the image classification_conv checkpoint also has batchnorm states (running_mean and running_var)
        params = checkpoint["params"]
        state = checkpoint["state"]
    else:
        params = checkpoint

    # turn into initial state dict
    state_dict = {}
    for scope_name, parameters in hk.data_structures.to_mutable_dict(params).items():
        for param_name, param in parameters.items():
            state_dict[scope_name + "/" + param_name] = param

    if state is not None:
        # add state variables
        for scope_name, parameters in hk.data_structures.to_mutable_dict(state).items():
            for param_name, param in parameters.items():
                state_dict[scope_name + "/" + param_name] = param

    # rename keys
    rename_keys(state_dict, architecture=architecture)

    # load HuggingFace model
    config = PerceiverConfig()
    subsampling = None
    repo_id = "huggingface/label-files"
    if architecture == "MLM":
        config.qk_channels = 8 * 32
        config.v_channels = 1280
        model = PerceiverForMaskedLM(config)
    elif "image_classification" in architecture:
        config.num_latents = 512
        config.d_latents = 1024
        config.d_model = 512
        config.num_blocks = 8
        config.num_self_attends_per_block = 6
        config.num_cross_attention_heads = 1
        config.num_self_attention_heads = 8
        config.qk_channels = None
        config.v_channels = None
        # set labels
        config.num_labels = 1000
        filename = "imagenet-1k-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
        if architecture == "image_classification":
            config.image_size = 224
            model = PerceiverForImageClassificationLearned(config)
        elif architecture == "image_classification_fourier":
            config.d_model = 261
            model = PerceiverForImageClassificationFourier(config)
        elif architecture == "image_classification_conv":
            config.d_model = 322
            model = PerceiverForImageClassificationConvProcessing(config)
        else:
            raise ValueError(f"Architecture {architecture} not supported")
    elif architecture == "optical_flow":
        config.num_latents = 2048
        config.d_latents = 512
        config.d_model = 322
        config.num_blocks = 1
        config.num_self_attends_per_block = 24
        config.num_self_attention_heads = 16
        config.num_cross_attention_heads = 1
        model = PerceiverForOpticalFlow(config)
    elif architecture == "multimodal_autoencoding":
        config.num_latents = 28 * 28 * 1
        config.d_latents = 512
        config.d_model = 704
        config.num_blocks = 1
        config.num_self_attends_per_block = 8
        config.num_self_attention_heads = 8
        config.num_cross_attention_heads = 1
        config.num_labels = 700
        # define dummy inputs + subsampling (as each forward pass is only on a chunk of image + audio data)
        images = torch.randn((1, 16, 3, 224, 224))
        audio = torch.randn((1, 30720, 1))
        nchunks = 128
        image_chunk_size = np.prod((16, 224, 224)) // nchunks
        audio_chunk_size = audio.shape[1] // config.samples_per_patch // nchunks
        # process the first chunk
        chunk_idx = 0
        subsampling = {
            "image": torch.arange(image_chunk_size * chunk_idx, image_chunk_size * (chunk_idx + 1)),
            "audio": torch.arange(audio_chunk_size * chunk_idx, audio_chunk_size * (chunk_idx + 1)),
            "label": None,
        }
        model = PerceiverForMultimodalAutoencoding(config)
        # set labels
        filename = "kinetics700-id2label.json"
        id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset"), "r"))
        id2label = {int(k): v for k, v in id2label.items()}
        config.id2label = id2label
        config.label2id = {v: k for k, v in id2label.items()}
    else:
        raise ValueError(f"Architecture {architecture} not supported")
    model.eval()

    # load weights
    model.load_state_dict(state_dict)

    # prepare dummy input
    input_mask = None
    if architecture == "MLM":
        tokenizer = PerceiverTokenizer.from_pretrained("/Users/NielsRogge/Documents/Perceiver/Tokenizer files")
        text = "This is an incomplete sentence where some words are missing."
        encoding = tokenizer(text, padding="max_length", return_tensors="pt")
        # mask " missing.". Note that the model performs much better if the masked chunk starts with a space.
        encoding.input_ids[0, 51:60] = tokenizer.mask_token_id
        inputs = encoding.input_ids
        input_mask = encoding.attention_mask
    elif architecture in ["image_classification", "image_classification_fourier", "image_classification_conv"]:
        image_processor = PerceiverImageProcessor()
        image = prepare_img()
        encoding = image_processor(image, return_tensors="pt")
        inputs = encoding.pixel_values
    elif architecture == "optical_flow":
        inputs = torch.randn(1, 2, 27, 368, 496)
    elif architecture == "multimodal_autoencoding":
        images = torch.randn((1, 16, 3, 224, 224))
        audio = torch.randn((1, 30720, 1))
        inputs = {"image": images, "audio": audio, "label": torch.zeros((images.shape[0], 700))}

    # forward pass
    if architecture == "multimodal_autoencoding":
        outputs = model(inputs=inputs, attention_mask=input_mask, subsampled_output_points=subsampling)
    else:
        outputs = model(inputs=inputs, attention_mask=input_mask)
    logits = outputs.logits

    # verify logits
    if not isinstance(logits, dict):
        print("Shape of logits:", logits.shape)
    else:
        for k, v in logits.items():
            print(f"Shape of logits of modality {k}", v.shape)

    if architecture == "MLM":
        expected_slice = torch.tensor(
            [[-11.8336, -11.6850, -11.8483], [-12.8149, -12.5863, -12.7904], [-12.8440, -12.6410, -12.8646]]
        )
        assert torch.allclose(logits[0, :3, :3], expected_slice)
        masked_tokens_predictions = logits[0, 51:60].argmax(dim=-1).tolist()
        expected_list = [38, 115, 111, 121, 121, 111, 116, 109, 52]
        assert masked_tokens_predictions == expected_list
        print("Greedy predictions:")
        print(masked_tokens_predictions)
        print()
        print("Predicted string:")
        print(tokenizer.decode(masked_tokens_predictions))

    elif architecture in ["image_classification", "image_classification_fourier", "image_classification_conv"]:
        print("Predicted class:", model.config.id2label[logits.argmax(-1).item()])

    # Finally, save files
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--pickle_file",
        type=str,
        default=None,
        required=True,
        help="Path to local pickle file of a Perceiver checkpoint you'd like to convert.\n"
        "Given the files are in the pickle format, please be wary of passing it files you trust.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default=None,
        type=str,
        required=True,
        help="Path to the output PyTorch model directory, provided as a string.",
    )
    parser.add_argument(
        "--architecture",
        default="MLM",
        type=str,
        help="""
        Architecture, provided as a string. One of 'MLM', 'image_classification', image_classification_fourier',
        image_classification_fourier', 'optical_flow' or 'multimodal_autoencoding'.
        """,
    )

    args = parser.parse_args()
    convert_perceiver_checkpoint(args.pickle_file, args.pytorch_dump_folder_path, args.architecture)
