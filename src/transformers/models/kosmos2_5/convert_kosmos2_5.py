import argparse

from fairseq.checkpoint_utils import load_checkpoint_to_cpu

from transformers import Kosmos2_5Config, Kosmos2_5ForConditionalGeneration


KEYS_TO_MODIFY_MAPPING = {
    "gpt_model.decoder.output_projection": "text_model.lm_head",
    "gpt_model.decoder": "text_model.model",
    "img_connector": "image_to_text_projection",
    "img_model.embeddings": "vision_model.embeddings",
    "img_model.encoder": "vision_model.encoder",
    "img_model.layernorm": "vision_model.layernorm",
    "img_model": "vision_model",
    "ln_pre": "pre_layrnorm",
    "ln_post": "post_layernorm",
    "transformer.resblocks": "encoder.layers",
    "ts_attn": "self_attn",
    "ln_1": "layer_norm1",
    "ln_2": "layer_norm2",
    "c_fc": "fc1",
    "c_proj": "fc2",
}


KEYS_TO_IGNORE = [
    # this buffer in the original code is only used to send weights to the desired device
    "gpt_model.decoder.embed_positions._float_tensor",
    # this weight is never used in the forward in the original KOSMOS-2.5)
    "gpt_model.decoder.self_attn_sope.scale",
]


def rename_key(key):
    for key_to_modify, new_key in KEYS_TO_MODIFY_MAPPING.items():
        if key_to_modify in key:
            key = key.replace(key_to_modify, new_key)

    return key


def convert_kosmos2_5_checkpoint_to_pytorch(checkpoint_path, pytorch_dump_folder_path):
    state = load_checkpoint_to_cpu(checkpoint_path)
    state_dict = state["model"]
    state_dict_keys = list(state_dict.keys())

    config = Kosmos2_5Config()
    # This is necessary to match the results given by the original demo
    config.text_config.no_repeat_ngram_size = 3
    model = Kosmos2_5ForConditionalGeneration(config)

    # convert (by renaming keys)
    converted_state_dict = {}
    for key in state_dict_keys:
        if key in KEYS_TO_IGNORE:
            continue
        renamed_key = rename_key(key)
        converted_state_dict[renamed_key] = state_dict[key]

    # set
    # check weight loading
    # check whether the state in converted_state_dict is the same as the state in the model
    model.load_state_dict(converted_state_dict, strict=True)
    # save the result
    model.save_pretrained(pytorch_dump_folder_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--kosmos2_5_checkpoint_path",
        default="ckpt.pt",
        type=str,
        required=False,
        help="Path the official PyTorch dump.",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        default="ckpt",
        type=str,
        required=False,
        help="Path to the output PyTorch model.",
    )
    args = parser.parse_args()
    convert_kosmos2_5_checkpoint_to_pytorch(args.kosmos2_5_checkpoint_path, args.pytorch_dump_folder_path)
