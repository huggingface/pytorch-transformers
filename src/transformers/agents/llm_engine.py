#!/usr/bin/env python
# coding=utf-8

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
from copy import deepcopy
from enum import Enum
from typing import Dict, List, Optional

from huggingface_hub import InferenceClient

from .. import AutoTokenizer
from ..pipelines.base import Pipeline
from ..utils import logging


logger = logging.get_logger(__name__)


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool-call"
    TOOL_RESPONSE = "tool-response"

    @classmethod
    def roles(cls):
        return [r.value for r in cls]


def get_clean_message_list(message_list: List[Dict[str, str]], role_conversions: Dict[str, str] = {}):
    """
    Subsequent messages with the same role will be concatenated to a single message.

    Args:
        message_list (`List[Dict[str, str]]`): List of chat messages.
    """
    final_message_list = []
    message_list = deepcopy(message_list)  # Avoid modifying the original list
    for message in message_list:
        if not set(message.keys()) == {"role", "content"}:
            raise ValueError("Message should contain only 'role' and 'content' keys!")

        role = message["role"]
        if role not in MessageRole.roles():
            raise ValueError(f"Incorrect role {role}, only {MessageRole.roles()} are supported for now.")

        if role in role_conversions:
            message["role"] = role_conversions[role]

        if len(final_message_list) > 0 and message["role"] == final_message_list[-1]["role"]:
            final_message_list[-1]["content"] += "\n=======\n" + message["content"]
        else:
            final_message_list.append(message)
    return final_message_list


llama_role_conversions = {
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}


class HfEngine:
    def __init__(self, model_id: Optional[str] = None):
        self.last_input_token_count = None
        self.last_output_token_count = None
        if model_id is None:
            model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
            logger.warning(f"Using default model for token counting: '{model_id}'")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer for model {model_id}: {e}. Loading default tokenizer instead.")
            self.tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")

    def get_token_counts(self):
        return {
            "input_token_count": self.last_input_token_count,
            "output_token_count": self.last_output_token_count,
        }

    def generate(
        self, messages: List[Dict[str, str]], stop_sequences: Optional[List[str]] = None, grammar: Optional[str] = None
    ):
        raise NotImplementedError

    def __call__(
        self, messages: List[Dict[str, str]], stop_sequences: Optional[List[str]] = None, grammar: Optional[str] = None
    ) -> str:
        """Process the input messages and return the model's response.

        This method sends a list of messages to the Hugging Face Inference API, optionally with stop sequences and grammar customization.

        Parameters:
            messages (`List[Dict[str, str]]`):
                A list of message dictionaries to be processed. Each dictionary should have the structure `{"role": "user/system", "content": "message content"}`.
            stop_sequences (`List[str]`, *optional*):
                A list of strings that will stop the generation if encountered in the model's output.
            grammar (`str`, *optional*):
                The grammar or formatting structure to use in the model's response.

        Returns:
            `str`: The text content of the model's response.

        Example:
            ```python
            >>> engine = HfApiEngine(
            ...     model="meta-llama/Meta-Llama-3.1-8B-Instruct",
            ...     token="your_hf_token_here",
            ...     max_tokens=2000
            ... )
            >>> messages = [{"role": "user", "content": "Explain quantum mechanics in simple terms."}]
            >>> response = engine(messages, stop_sequences=["END"])
            >>> print(response)
            "Quantum mechanics is the branch of physics that studies..."
            ```
        """
        if not isinstance(messages, List):
            raise ValueError("Messages should be a list of dictionaries with 'role' and 'content' keys.")
        if stop_sequences is None:
            stop_sequences = []
        response = self.generate(messages, stop_sequences, grammar)
        self.last_input_token_count = len(self.tokenizer.apply_chat_template(messages, tokenize=True))
        self.last_output_token_count = len(self.tokenizer.encode(response))

        # Remove stop sequences from LLM output
        for stop_seq in stop_sequences:
            if response[-len(stop_seq) :] == stop_seq:
                response = response[: -len(stop_seq)]
        return response


class HfApiEngine(HfEngine):
    """A class to interact with Hugging Face's Inference API for language model interaction.

    This engine allows you to communicate with Hugging Face's models using the Inference API. It can be used in both serverless mode or with a dedicated endpoint, supporting features like stop sequences and grammar customization.

    Parameters:
        model (`str`, *optional*, defaults to `"meta-llama/Meta-Llama-3.1-8B-Instruct"`):
            The Hugging Face model ID to be used for inference. This can be a path or model identifier from the Hugging Face model hub.
        token (`str`, *optional*):
            Token used by the Hugging Face API for authentication.
            If not provided, the class will use the token stored in the Hugging Face CLI configuration.
        max_tokens (`int`, *optional*, defaults to 1500):
            The maximum number of tokens allowed in the output.
        timeout (`int`, *optional*, defaults to 120):
            Timeout for the API request, in seconds.

    Raises:
        ValueError:
            If the model name is not provided.
    """

    def __init__(
        self,
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        token: Optional[str] = None,
        max_tokens: Optional[int] = 1500,
        timeout: Optional[int] = 120,
    ):
        super().__init__(model_id=model)
        self.model = model
        self.client = InferenceClient(self.model, token=token, timeout=timeout)
        self.max_tokens = max_tokens

    def generate(
        self, messages: List[Dict[str, str]], stop_sequences: Optional[List[str]] = None, grammar: Optional[str] = None
    ) -> str:
        # Get clean message list
        messages = get_clean_message_list(messages, role_conversions=llama_role_conversions)

        # Send messages to the Hugging Face Inference API
        if grammar is not None:
            response = self.client.chat_completion(
                messages, stop=stop_sequences, max_tokens=self.max_tokens, response_format=grammar
            )
        else:
            response = self.client.chat_completion(messages, stop=stop_sequences, max_tokens=self.max_tokens)

        response = response.choices[0].message.content
        return response


class TransformersEngine(HfEngine):
    """This engine uses a pre-initialized local text-generation pipeline."""

    def __init__(self, pipeline: Pipeline, model_id: Optional[str] = None):
        super().__init__(model_id)
        self.pipeline = pipeline

    def generate(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        max_length: int = 1500,
    ) -> str:
        # Get clean message list
        messages = get_clean_message_list(messages, role_conversions=llama_role_conversions)

        # Get LLM output
        if stop_sequences is not None and len(stop_sequences) > 0:
            stop_strings = stop_sequences
        else:
            stop_strings = None

        output = self.pipeline(
            messages,
            stop_strings=stop_strings,
            max_length=max_length,
            tokenizer=self.pipeline.tokenizer,
        )

        response = output[0]["generated_text"][-1]["content"]
        return response


DEFAULT_JSONAGENT_REGEX_GRAMMAR = {
    "type": "regex",
    "value": 'Thought: .+?\\nAction:\\n\\{\\n\\s{4}"action":\\s"[^"\\n]+",\\n\\s{4}"action_input":\\s"[^"\\n]+"\\n\\}\\n<end_action>',
}

DEFAULT_CODEAGENT_REGEX_GRAMMAR = {
    "type": "regex",
    "value": "Thought: .+?\\nCode:\\n```(?:py|python)?\\n(?:.|\\s)+?\\n```<end_action>",
}
