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
import glob
import json
import os
from typing import List, Optional

import regex as re
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from transformers import (
    GenerationConfig,
    GotOcr2Config,
    GotOcr2ForConditionalGeneration,
    GotOcr2ImageProcessor,
    GotOcr2Processor,
    PreTrainedTokenizerFast,
    is_vision_available,
)
from transformers.convert_slow_tokenizer import TikTokenConverter
from transformers.tokenization_utils import AddedToken


if is_vision_available():
    from transformers.image_utils import load_image


# fmt: off
ORIGINAL_TO_CONVERTED_KEY_MAPPING = {
    # Vision encoder mapping
    r"model.vision_tower_high.pos_embed":                           r"visual.pos_embed",
    r"model.vision_tower_high.patch_embed.proj":                    r"visual.patch_embed.projection",
    r"model.vision_tower_high.blocks.(\d+).norm":                   r"visual.layers.\1.layer_norm",
    r"model.vision_tower_high.blocks.(\d+).attn":                   r"visual.layers.\1.attn",
    r"model.vision_tower_high.blocks.(\d+).mlp":                    r"visual.layers.\1.mlp",
    r"model.vision_tower_high.neck.0":                              r"visual.neck.conv1",
    r"model.vision_tower_high.neck.1":                              r"visual.neck.layer_norm1",
    r"model.vision_tower_high.neck.2":                              r"visual.neck.conv2",
    r"model.vision_tower_high.neck.3":                              r"visual.neck.layer_norm2",
    r"model.vision_tower_high.net_(\d+)":                           lambda m: f"visual_adapter.conv_up{int(m.group(1)) - 1}",
    r"model.mm_projector_vary" :                                    r"visual_adapter.multimodal_projector",
}
# fmt: on

CONTEXT_LENGTH = 8000


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
            new_text = re.sub(pattern, replacement, new_text)
        output_dict = dict(zip(old_text.split("\n"), new_text.split("\n")))
    return output_dict


def load_original_state_dict(model_id):
    directory_path = snapshot_download(repo_id=model_id, allow_patterns=["*.safetensors"])

    original_state_dict = {}
    for path in glob.glob(f"{directory_path}/*"):
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    original_state_dict[key] = f.get_tensor(key)

    return original_state_dict


def get_got_ocr2_config():
    config = GotOcr2Config(
        vocab_size=151860,
        hidden_size=1024,
        intermediate_size=2816,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=16,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=True,
        tie_word_embeddings=True,
        rope_theta=1000000.0,
        use_sliding_window=False,
        sliding_window=32768,
        max_window_layers=21,
        attention_dropout=0.0,
        rope_scaling=None,
        image_token_id=151859,
    )

    return config


def write_model(
    model_path,
    input_base_path,
    instruct=False,
    push_to_hub=False,
):
    os.makedirs(model_path, exist_ok=True)

    config = get_got_ocr2_config()
    config.architectures = ["GotOcr2ForConditionalGeneration"]
    config.save_pretrained(model_path)
    print("Model config saved successfully...")

    # ------------------------------------------------------------
    # Convert weights
    # ------------------------------------------------------------

    print(f"Fetching all parameters from the checkpoint at {input_base_path}...")
    state_dict_old = load_original_state_dict(input_base_path)
    print("Converting model...")
    all_keys = list(state_dict_old.keys())
    new_keys = convert_old_keys_to_new_keys(all_keys)
    state_dict = {}
    for key in all_keys:
        new_key = new_keys[key]
        state_dict[new_key] = state_dict_old[key]

    del state_dict_old
    gc.collect()

    print("Loading the checkpoint in a GotOcr2ForConditionalGeneration model.")
    model = GotOcr2ForConditionalGeneration(config)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    model = model.to(torch.bfloat16)
    print("model dtype:", model.dtype)
    print("Missing keys:", missing_keys)
    print("Unexpected keys:", unexpected_keys)

    print("Saving the model.")
    model.save_pretrained(model_path)
    if push_to_hub:
        model.push_to_hub("yonigozlan/GOT-OCR-2.0-hf", use_temp_dir=True)
    del state_dict, model

    # Safety check: reload the converted model
    gc.collect()
    print("Reloading the model to check if it's saved correctly.")
    model = GotOcr2ForConditionalGeneration.from_pretrained(model_path, device_map="auto")
    processor = GotOcr2Processor.from_pretrained(model_path)
    image = load_image(
        "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg"
    )

    inputs = processor(image, return_tensors="pt", format=True).to(model.device, dtype=model.dtype)
    generate_ids = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=4)
    decoded_output = processor.decode(generate_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    expected_output = "\\title{\nR"
    print("Decoded output:", decoded_output)
    assert decoded_output == expected_output
    print("Model reloaded successfully.")
    del model

    # generation config
    if instruct:
        print("Saving generation config...")
        generation_config = GenerationConfig(
            bos_token_id=151643,
            eos_token_id=151643,
            max_new_tokens=2048,
        )
        generation_config.save_pretrained(model_path)


class GotOcr2Converter(TikTokenConverter):
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


def write_tokenizer(tokenizer_path: str, save_dir: str, instruct: bool = False, push_to_hub: bool = False):
    model_max_length = CONTEXT_LENGTH
    pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: W605
    # Special tokens
    special_tokens = (
        ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
        + [f"<|extra_{i}|>" for i in range(205)]
        + [
            "<ref>",
            "</ref>",
            "<box>",
            "</box>",
            "<quad>",
            "</quad>",
            "<img>",
            "</img>",
            "<imgpad>",
        ]
    )

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
    pad_token = "<|endoftext|>"
    pad_token = AddedToken(pad_token, lstrip=False, rstrip=False, normalized=False, single_word=False)

    converter = GotOcr2Converter(
        vocab_file=tokenizer_path,
        pattern=pattern,
        special_tokens=special_tokens,
        model_max_length=model_max_length,
        chat_template=chat_template if instruct else None,
        pad_token=pad_token,
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        clean_up_tokenization_spaces=True,
    )
    tokenizer = converter.tokenizer
    tokenizer.save_pretrained(save_dir)

    if push_to_hub:
        tokenizer.push_to_hub("yonigozlan/GOT-OCR-2.0-hf", use_temp_dir=True)

    if instruct:
        print("Saving chat template...")
        chat_template_path = os.path.join(save_dir, "chat_template.json")
        with open(chat_template_path, "w") as f:
            json.dump({"chat_template": chat_template}, f, indent=2)


def write_image_processor(save_dir: str, push_to_hub: bool = False):
    image_processor = GotOcr2ImageProcessor(
        do_resize=True,
        size={"height": 1024, "width": 1024},
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
    )

    image_processor.save_pretrained(save_dir)
    if push_to_hub:
        image_processor.push_to_hub("yonigozlan/GOT-OCR-2.0-hf", use_temp_dir=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="stepfun-ai/GOT-OCR2_0",
        help="Location of LLaMA weights, which contains tokenizer.model and model folders",
    )
    parser.add_argument(
        "--output_dir",
        default="GotOcr2",
        help="Location to write HF model and tokenizer",
    )

    parser.add_argument(
        "--instruct",
        action="store_true",
        help="Whether the model is an instruct model",
    )
    parser.add_argument(
        "--push_to_hub", action="store_true", help="Whether or not to push the converted model to the 🤗 hub."
    )
    args = parser.parse_args()
    write_tokenizer(
        tokenizer_path="qwen.tiktoken",
        save_dir=args.output_dir,
        instruct=args.instruct,
        push_to_hub=args.push_to_hub,
    )

    write_image_processor(
        save_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
    )
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        instruct=args.instruct,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
