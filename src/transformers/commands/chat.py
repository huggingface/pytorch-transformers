# Copyright 2025 The HuggingFace Team. All rights reserved.
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


import copy
import json
import os
import platform
import re
import time
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from threading import Thread
from typing import Optional

import torch
import yaml
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

from . import BaseTransformersCLICommand


if platform.system() != "Windows":
    import pwd


HELP_STRING = """\

**TRANSFORMERS CHAT INTERFACE**

The chat interface is a simple tool to try out a chat model.

Besides talking to the model there are several commands:
- **help**: show this help message
- **clear**: clears the current conversation and start a new one
- **example {NAME}**: load example named `{NAME}` from the config and use it as the user input
- **set {SETTING_NAME}={SETTING_VALUE};**: change the system prompt or generation settings (multiple settings are separated by a ';').
- **reset**: same as clear but also resets the generation configs to defaults if they have been changed by **set**
- **save {SAVE_NAME} (optional)**: save the current chat and settings to file by default to `./chat_history/{MODEL_NAME}/chat_{DATETIME}.yaml` or `{SAVE_NAME}` if provided
- **exit**: closes the interface
"""

SUPPORTED_GENERATION_KWARGS = [
    "max_new_tokens",
    "do_sample",
    "num_beams",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
]

SETTING_RE = r"^set\s+[A-Za-z\s_]+=[A-Za-z\d\s.!\"#$%&'()*+,-/:<=>?@\[\]^_`{|}~]+(?:;\s*[A-Za-z\s_]+=[A-Za-z\d\s.!\"#$%&'()*+,-/:<=>?@\[\]^_`{|}~]+)*$"

DEFAULT_EXAMPLES = {
    "llama": {"text": "There is a Llama in my lawn, how can I get rid of it?"},
    "code": {
        "text": "Write a Python function that integrates any Python function f(x) numerically over an arbitrary interval [x_start, x_end]."
    },
    "helicopter": {"text": "How many helicopters can a human eat in one sitting?"},
    "numbers": {"text": "Count to 10 but skip every number ending with an 'e'"},
    "birds": {"text": "Why aren't birds real?"},
    "socks": {"text": "Why is it important to eat socks after meditating?"},
}


def get_username():
    if platform.system() == "Windows":
        return os.getlogin()
    else:
        return pwd.getpwuid(os.getuid()).pw_name


def create_default_filename(model_name):
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    return f"{model_name}/chat_{time_str}.json"


def save_chat(chat, args, filename):
    output_dict = {}
    output_dict["settings"] = vars(args)
    output_dict["chat_history"] = chat

    folder = args.save_folder

    if filename is None:
        filename = create_default_filename(args.model_name_or_path)
        filename = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as f:
        json.dump(output_dict, f, indent=4)
    return os.path.abspath(filename)


def clear_chat_history(system_prompt):
    if system_prompt is None:
        chat = []
    else:
        chat = [{"role": "system", "content": system_prompt}]
    return chat


def parse_settings(user_input, current_args, interface):
    settings = user_input[4:].strip().split(";")
    settings = [(setting.split("=")[0], setting[len(setting.split("=")[0]) + 1 :]) for setting in settings]
    settings = dict(settings)
    error = False

    for name in settings:
        if hasattr(current_args, name):
            try:
                if isinstance(getattr(current_args, name), bool):
                    if settings[name] == "True":
                        settings[name] = True
                    elif settings[name] == "False":
                        settings[name] = False
                    else:
                        raise ValueError
                else:
                    settings[name] = type(getattr(current_args, name))(settings[name])
            except ValueError:
                interface.print_red(
                    f"Cannot cast setting {name} (={settings[name]}) to {type(getattr(current_args, name))}."
                )
        else:
            interface.print_red(f"There is no '{name}' setting.")

    if error:
        interface.print_red("There was an issue parsing the settings. No settings have been changed.")
        return current_args, False
    else:
        for name in settings:
            setattr(current_args, name, settings[name])
            interface.print_green(f"Set {name} to {settings[name]}.")

        time.sleep(1.5)  # so the user has time to read the changes
        return current_args, True


def get_quantization_config(model_args) -> Optional[BitsAndBytesConfig]:
    if model_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=model_args.torch_dtype,  # For consistency with model weights, we use the same value as `torch_dtype`
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=model_args.use_bnb_nested_quant,
            bnb_4bit_quant_storage=model_args.torch_dtype,
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
    else:
        quantization_config = None

    return quantization_config


def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        revision=args.model_revision,
        trust_remote_code=args.trust_remote_code,
    )

    torch_dtype = args.torch_dtype if args.torch_dtype in ["auto", None] else getattr(torch, args.torch_dtype)
    quantization_config = get_quantization_config(args)
    model_kwargs = {
        "revision": args.model_revision,
        "attn_implementation": args.attn_implementation,
        "torch_dtype": torch_dtype,
        "device_map": "auto",
        "quantization_config": quantization_config,
    }
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, trust_remote_code=args.trust_remote_code, **model_kwargs
    )

    if getattr(model, "hf_device_map", None) is None:
        model = model.to(args.device)

    return model, tokenizer


def parse_eos_tokens(tokenizer, eos_tokens, eos_token_ids):
    if tokenizer.pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    else:
        pad_token_id = tokenizer.pad_token_id

    all_eos_token_ids = []

    if eos_tokens is not None:
        all_eos_token_ids.extend(tokenizer.convert_tokens_to_ids(eos_tokens.split(",")))

    if eos_token_ids is not None:
        all_eos_token_ids.extend([int(token_id) for token_id in eos_token_ids.split(",")])

    if len(all_eos_token_ids) == 0:
        all_eos_token_ids.append(tokenizer.eos_token_id)

    return pad_token_id, all_eos_token_ids


class RichInterface:
    def __init__(self, model_name=None, user_name=None):
        self._console = Console()
        if model_name is None:
            self.model_name = "assistant"
        else:
            self.model_name = model_name
        if user_name is None:
            self.user_name = "user"
        else:
            self.user_name = user_name

    def stream_output(self, output_stream):
        """Stream output from a role."""
        # This method is originally from the FastChat CLI: https://github.com/lm-sys/FastChat/blob/main/fastchat/serve/cli.py
        # Create a Live context for updating the console output
        text = ""
        self._console.print(f"[bold blue]<{self.model_name}>:")
        with Live(console=self._console, refresh_per_second=4) as live:
            # Read lines from the stream
            for i, outputs in enumerate(output_stream):
                if not outputs or i == 0:
                    continue
                text += outputs
                # Render the accumulated text as Markdown
                # NOTE: this is a workaround for the rendering "unstandard markdown"
                #  in rich. The chatbots output treat "\n" as a new line for
                #  better compatibility with real-world text. However, rendering
                #  in markdown would break the format. It is because standard markdown
                #  treat a single "\n" in normal text as a space.
                #  Our workaround is adding two spaces at the end of each line.
                #  This is not a perfect solution, as it would
                #  introduce trailing spaces (only) in code block, but it works well
                #  especially for console output, because in general the console does not
                #  care about trailing spaces.
                lines = []
                for line in text.splitlines():
                    lines.append(line)
                    if line.startswith("```"):
                        # Code block marker - do not add trailing spaces, as it would
                        #  break the syntax highlighting
                        lines.append("\n")
                    else:
                        lines.append("  \n")
                markdown = Markdown("".join(lines).strip(), code_theme="github-dark")
                # Update the Live console output
                live.update(markdown)
        self._console.print()
        return text

    def input(self):
        input = self._console.input(f"[bold red]<{self.user_name}>:\n")
        self._console.print()
        return input

    def clear(self):
        self._console.clear()

    def print_user_message(self, text):
        self._console.print(f"[bold red]<{self.user_name}>:[/ bold red]\n{text}")
        self._console.print()

    def print_green(self, text):
        self._console.print(f"[bold green]{text}")
        self._console.print()

    def print_red(self, text):
        self._console.print(f"[bold red]{text}")
        self._console.print()

    def print_help(self):
        self._console.print(Markdown(HELP_STRING))
        self._console.print()


@dataclass
class ChatArguments:
    r"""
    Arguments for the chat script.

    Args:
        model_name_or_path (`str`):
            Name of the pre-trained model.
        user (`str` or `None`, *optional*, defaults to `None`):
            Username to display in chat interface.
        system_prompt (`str` or `None`, *optional*, defaults to `None`):
            System prompt.
        save_folder (`str`, *optional*, defaults to `"./chat_history/"`):
            Folder to save chat history.
        device (`str`, *optional*, defaults to `"cpu"`):
            Device to use for inference.
        examples_path (`str` or `None`, *optional*, defaults to `None`):
            Path to a yaml file with examples.
        max_new_tokens (`int`, *optional*, defaults to `256`):
            Maximum number of tokens to generate.
        do_sample (`bool`, *optional*, defaults to `True`):
            Whether to sample outputs during generation.
        num_beams (`int`, *optional*, defaults to `1`):
            Number of beams for beam search.
        temperature (`float`, *optional*, defaults to `1.0`):
            Temperature parameter for generation.
        top_k (`int`, *optional*, defaults to `50`):
            Value of k for top-k sampling.
        top_p (`float`, *optional*, defaults to `1.0`):
            Value of p for nucleus sampling.
        repetition_penalty (`float`, *optional*, defaults to `1.0`):
            Repetition penalty.
        eos_tokens (`str` or `None`, *optional*, defaults to `None`):
            EOS tokens to stop the generation. If multiple they should be comma separated.
        eos_token_ids (`str` or `None`, *optional*, defaults to `None`):
            EOS token IDs to stop the generation. If multiple they should be comma separated.
        model_revision (`str`, *optional*, defaults to `"main"`):
            Specific model version to use (can be a branch name, tag name or commit id).
        torch_dtype (`str` or `None`, *optional*, defaults to `None`):
            Override the default `torch.dtype` and load the model under this dtype. If `'auto'` is passed, the dtype
            will be automatically derived from the model's weights.
        trust_remote_code (`bool`, *optional*, defaults to `False`):
            Whether to trust remote code when loading a model.
        attn_implementation (`str` or `None`, *optional*, defaults to `None`):
            Which attention implementation to use; you can run --attn_implementation=flash_attention_2, in which case
            you must install this manually by running `pip install flash-attn --no-build-isolation`.
        load_in_8bit (`bool`, *optional*, defaults to `False`):
            Whether to use 8 bit precision for the base model - works only with LoRA.
        load_in_4bit (`bool`, *optional*, defaults to `False`):
            Whether to use 4 bit precision for the base model - works only with LoRA.
        bnb_4bit_quant_type (`str`, *optional*, defaults to `"nf4"`):
            Quantization type.
        use_bnb_nested_quant (`bool`, *optional*, defaults to `False`):
            Whether to use nested quantization.
    """

    # General settings
    model_name_or_path: str = field(metadata={"help": "Name of the pre-trained model."})
    user: Optional[str] = field(default=None, metadata={"help": "Username to display in chat interface."})
    system_prompt: Optional[str] = field(default=None, metadata={"help": "System prompt."})
    save_folder: str = field(default="./chat_history/", metadata={"help": "Folder to save chat history."})
    device: str = field(default="cpu", metadata={"help": "Device to use for inference."})
    examples_path: Optional[str] = field(default=None, metadata={"help": "Path to a yaml file with examples."})

    # Generation settings
    max_new_tokens: int = field(default=256, metadata={"help": "Maximum number of tokens to generate."})
    do_sample: bool = field(default=True, metadata={"help": "Whether to sample outputs during generation."})
    num_beams: int = field(default=1, metadata={"help": "Number of beams for beam search."})
    temperature: float = field(default=1.0, metadata={"help": "Temperature parameter for generation."})
    top_k: int = field(default=50, metadata={"help": "Value of k for top-k sampling."})
    top_p: float = field(default=1.0, metadata={"help": "Value of p for nucleus sampling."})
    repetition_penalty: float = field(default=1.0, metadata={"help": "Repetition penalty."})
    eos_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "EOS tokens to stop the generation. If multiple they should be comma separated."},
    )
    eos_token_ids: Optional[str] = field(
        default=None,
        metadata={"help": "EOS token IDs to stop the generation. If multiple they should be comma separated."},
    )

    # Model loading
    model_revision: str = field(
        default="main",
        metadata={"help": "Specific model version to use (can be a branch name, tag name or commit id)."},
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override the default `torch.dtype` and load the model under this dtype. If `'auto'` is passed, "
            "the dtype will be automatically derived from the model's weights.",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    trust_remote_code: bool = field(
        default=False, metadata={"help": "Whether to trust remote code when loading a model."}
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which attention implementation to use; you can run --attn_implementation=flash_attention_2, in "
            "which case you must install this manually by running `pip install flash-attn --no-build-isolation`."
        },
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 8 bit precision for the base model - works only with LoRA."},
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 4 bit precision for the base model - works only with LoRA."},
    )
    bnb_4bit_quant_type: str = field(default="nf4", metadata={"help": "Quantization type.", "choices": ["fp4", "nf4"]})
    use_bnb_nested_quant: bool = field(default=False, metadata={"help": "Whether to use nested quantization."})


def chat_command_factory(args: Namespace):
    """
    Factory function used to chat with a local model.
    """
    return ChatCommand(args)


class ChatCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        """
        Register this command to argparse so it's available for the transformer-cli

        Args:
            parser: Root parser to register command-specific arguments
        """
        dataclass_types = (ChatArguments,)
        chat_parser = parser.add_parser("chat", help=HELP_STRING, dataclass_types=dataclass_types)
        chat_parser.set_defaults(func=chat_command_factory)

    def __init__(self, args):
        self.args = args

    def run(self):
        args = self.args
        if args.examples_path is None:
            examples = DEFAULT_EXAMPLES
        else:
            with open(args.examples_path) as f:
                examples = yaml.safe_load(f)

        current_args = copy.deepcopy(args)

        if args.user is None:
            user = get_username()
        else:
            user = args.user

        model, tokenizer = load_model_and_tokenizer(args)
        generation_streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

        pad_token_id, eos_token_ids = parse_eos_tokens(tokenizer, args.eos_tokens, args.eos_token_ids)

        interface = RichInterface(model_name=args.model_name_or_path, user_name=user)
        interface.clear()
        chat = clear_chat_history(current_args.system_prompt)
        while True:
            try:
                user_input = interface.input()

                if user_input == "clear":
                    chat = clear_chat_history(current_args.system_prompt)
                    interface.clear()
                    continue

                if user_input == "help":
                    interface.print_help()
                    continue

                if user_input == "exit":
                    break

                if user_input == "reset":
                    interface.clear()
                    current_args = copy.deepcopy(args)
                    chat = clear_chat_history(current_args.system_prompt)
                    continue

                if user_input.startswith("save") and len(user_input.split()) < 2:
                    split_input = user_input.split()

                    if len(split_input) == 2:
                        filename = split_input[1]
                    else:
                        filename = None
                    filename = save_chat(chat, current_args, filename)
                    interface.print_green(f"Chat saved in {filename}!")
                    continue

                if re.match(SETTING_RE, user_input):
                    current_args, success = parse_settings(user_input, current_args, interface)
                    if success:
                        chat = []
                        interface.clear()
                        continue

                if user_input.startswith("example") and len(user_input.split()) == 2:
                    example_name = user_input.split()[1]
                    if example_name in examples:
                        interface.clear()
                        chat = []
                        interface.print_user_message(examples[example_name]["text"])
                        user_input = examples[example_name]["text"]
                    else:
                        interface.print_red(
                            f"Example {example_name} not found in list of available examples: {list(examples.keys())}."
                        )
                        continue

                chat.append({"role": "user", "content": user_input})

                inputs = tokenizer.apply_chat_template(chat, return_tensors="pt", add_generation_prompt=True).to(
                    model.device
                )
                attention_mask = torch.ones_like(inputs)
                generation_kwargs = {
                    "inputs": inputs,
                    "attention_mask": attention_mask,
                    "streamer": generation_streamer,
                    "max_new_tokens": current_args.max_new_tokens,
                    "do_sample": current_args.do_sample,
                    "num_beams": current_args.num_beams,
                    "temperature": current_args.temperature,
                    "top_k": current_args.top_k,
                    "top_p": current_args.top_p,
                    "repetition_penalty": current_args.repetition_penalty,
                    "pad_token_id": pad_token_id,
                    "eos_token_id": eos_token_ids,
                }

                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()
                model_output = interface.stream_output(generation_streamer)
                thread.join()
                chat.append({"role": "assistant", "content": model_output})

            except KeyboardInterrupt:
                break
