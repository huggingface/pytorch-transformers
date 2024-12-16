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
import glob

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from transformers import (
    AddedToken,
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoConfig,
    AutoTokenizer,
)


EPILOG_TXT = """Example:
    python transformers/src/transformers/models/aria/convert_aria_weights_to_hf.py --text_model_id rhymes-ai/Aria --vision_model_id rhymes-ai/Aria --output_hub_path m-ric/Aria_hf_2 --old_state_dict_id rhymes-ai/Aria

Example for creating the old state dict file with Python:

    import torch
    from aria.model.language_model.aria_llama import AriaTextForCausalLM

    # load model
    kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
    model = AriaTextForCausalLM.from_pretrained("rhymes-ai/Aria", low_cpu_mem_usage=True, **kwargs)

    # load vision tower
    model.get_vision_tower().load_model()

    # Save state dict
    torch.save(model.state_dict(), "tmp/hf_models/aria/model_state_dict.bin")
"""

KEYS_TO_MODIFY_MAPPING = {
    "vision_tower.vision_model": "vision_tower",
    "ln_ffn": "layer_norm",
    "ffn": "feed_forward",
    "ln_kv": "layer_norm_kv",
}


def load_original_state_dict(model_id):
    directory_path = snapshot_download(repo_id=model_id, allow_patterns=["*.safetensors"])

    original_state_dict = {}
    for path in glob.glob(f"{directory_path}/*"):
        if path.endswith(".safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    original_state_dict[key] = f.get_tensor(key)

    return original_state_dict


def convert_state_dict_to_hf(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.endswith(".inv_freq"):
            continue
        for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
            if key_to_modify in key:
                key = key.replace(key_to_modify, new_key)

        new_state_dict[key] = value
    new_state_dict["vision_tower.post_layernorm.weight"] = torch.zeros((1152,))
    new_state_dict["vision_tower.post_layernorm.bias"] = torch.zeros((1152,))

    return new_state_dict


def convert_aria_llama_to_hf(text_model_id, vision_model_id, output_hub_path, old_state_dict_id):
    torch.set_default_dtype(torch.float16)

    tokenizer = AutoTokenizer.from_pretrained(
        text_model_id,
        extra_special_tokens={
            "image_token": "<|img|>",
            "pad_token": "<pad>",
        },
    )
    tokenizer.add_tokens(AddedToken("<|img|>", special=True, normalized=False), special_tokens=True)
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}{% elif message['content'] is iterable %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% elif item['type'] == 'image' %}<fim_prefix><|img|><fim_suffix>{% endif %}{% endfor %}{% endif %}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

    processor = AriaProcessor.from_pretrained(
        text_model_id,
        tokenizer=tokenizer,
    )

    config = AutoConfig.from_pretrained(text_model_id)
    config.vision_config.hidden_size = 1152
    config.vision_config.attention_heads = 16
    config.pad_token_id = 2
    config.image_token_index = 9
    config.intermediate_size = config.moe_intermediate_size
    config.auto_map = {
        "AutoConfig": "modeling_aria.AriaConfig",
        "AutoModelForCausalLM": "modeling_aria.AriaForConditionalGeneration",
    }

    with torch.device("meta"):
        model = AriaForConditionalGeneration(config)

    state_dict = load_original_state_dict(old_state_dict_id)

    state_dict = convert_state_dict_to_hf(state_dict)
    model.load_state_dict(state_dict, strict=False, assign=True)

    # print("Saving models")
    # model.save_pretrained("local_aria", safe_serialization=False)
    # processor.save_pretrained("local_aria")
    print("Pushing to hub")
    model.push_to_hub(output_hub_path, create_pr=True)
    processor.push_to_hub(output_hub_path, create_pr=True)


def main():
    parser = argparse.ArgumentParser(
        epilog=EPILOG_TXT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--text_model_id",
        default="rhymes-ai/Aria",
        help="Hub location of the text model",
    )
    parser.add_argument(
        "--vision_model_id",
        default="rhymes-ai/Aria",
        help="Hub location of the vision model",
    )
    parser.add_argument(
        "--output_hub_path",
        default="rhymes-ai/Aria",
        help="Location on the hub of the converted model",
    )
    parser.add_argument(
        "--old_state_dict_id",
        default="rhymes-ai/Aria",
        help="Location on the hub of the raw state dict of the original model. The filename needs to be `model_state_dict.bin`",
    )
    args = parser.parse_args()
    convert_aria_llama_to_hf(args.text_model_id, args.vision_model_id, args.output_hub_path, args.old_state_dict_id)


if __name__ == "__main__":
    main()
