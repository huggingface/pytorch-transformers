# coding=utf-8
# Copyright 2024 Cohere team. All rights reserved.
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

# This file is based on the tokenization_llama_fast.py file in transformers


from typing import Dict, List, Literal, Optional, Union

from tokenizers import processors

from ...pipelines.conversational import Conversation
from ...tokenization_utils_base import TensorType
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging
from ...utils.versions import require_version


require_version("tokenizers>=0.13.3")

logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {"vocab_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "cohere-tokenizer": "https://huggingface.co/Cohere/Command-nightly/blob/main/tokenizer.json",
    },
}

# fmt: off
DEFAULT_SYSTEM_PROMPT = "You are Command-R, a brilliant, sophisticated, AI-assistant trained to assist human users by providing thorough responses. You are trained by Cohere."
DEFAULT_RAG_PREAMBLE = """## Task and Context
You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.

## Style Guide
Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling."""
# fmt: on


class CohereTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a Cohere tokenizer. Based on byte-level Byte-Pair-Encoding.

    This uses notably ByteFallback and NFC normalization.

    ```python
    >>> from transformers import AutoTokenizer

    >>> tokenizer = AutoTokenizer.from_pretrained("CohereForAI/c4ai-command-r-0.1")
    >>> tokenizer.encode("Hello this is a test")
    [1, 15043, 445, 338, 263, 1243]
    ```

    If you want to change the `bos_token` or the `eos_token`, make sure to specify them when initializing the model, or
    call `tokenizer.update_post_processor()` to make sure that the post-processing is correctly done (otherwise the
    values of the first token and final token of an encoded sequence will not be correct). For more details, checkout
    [post-processors] (https://huggingface.co/docs/tokenizers/api/post-processors) documentation.


    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`, *optional*):
            [SentencePiece](https://github.com/google/sentencepiece) file (generally has a .model extension) that
            contains the vocabulary necessary to instantiate a tokenizer.
        tokenizer_file (`str`, *optional*):
            [tokenizers](https://github.com/huggingface/tokenizers) file (generally has a .json extension) that
            contains everything needed to load the tokenizer.
        clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
            Whether or not to cleanup spaces after decoding, cleanup consists in removing potential artifacts like
            extra spaces.
        unk_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<UNK>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<BOS_TOKEN>"`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
        eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"<|END_OF_TURN_TOKEN|>"`):
            The end of sequence token.
        add_bos_token (`bool`, *optional*, defaults to `True`):
            Whether or not to add an `bos_token` at the start of sequences.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an `eos_token` at the end of sequences.
        use_default_system_prompt (`bool`, *optional*, defaults to `False`):
            Whether or not the default system prompt for Cohere tokenizer should be used.
        add_prefix_space (`bool`, *optional*):
            Whether or not the tokenizer should automatically add a prefix space
    """

    vocab_files_names = VOCAB_FILES_NAMES
    padding_side = "left"
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file=None,
        tokenizer_file=None,
        clean_up_tokenization_spaces=False,
        unk_token="<UNK>",
        bos_token="<BOS_TOKEN>",
        eos_token="<|END_OF_TURN_TOKEN|>",
        add_bos_token=True,
        add_eos_token=False,
        use_default_system_prompt=False,
        add_prefix_space=None,
        **kwargs,
    ):
        if add_prefix_space is not None:
            logger.warning_once(
                "You set `add_prefix_space`. The tokenizer needs to be converted from the slow tokenizers but Cohere tokenizer does not have a slow tokenizer. The `add_prefix_space` argument will be ignored."
            )

        super().__init__(
            vocab_file=vocab_file,
            tokenizer_file=tokenizer_file,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            use_default_system_prompt=use_default_system_prompt,
            **kwargs,
        )
        self._add_bos_token = add_bos_token
        self._add_eos_token = add_eos_token
        self.update_post_processor()
        self.use_default_system_prompt = use_default_system_prompt
        self.vocab_file = vocab_file
        self.grounded_generation_template = kwargs.pop("grounded_generation_template", None)
        self.tool_use_template = kwargs.pop("tool_use_template", None)

    def update_post_processor(self):
        """
        Updates the underlying post processor with the current `bos_token` and `eos_token`.
        """
        bos = self.bos_token
        bos_token_id = self.bos_token_id
        if bos is None and self.add_bos_token:
            raise ValueError("add_bos_token = True but bos_token = None")

        eos = self.eos_token
        eos_token_id = self.eos_token_id
        if eos is None and self.add_eos_token:
            raise ValueError("add_eos_token = True but eos_token = None")

        single = f"{(bos+':0 ') if self.add_bos_token else ''}$A:0{(' '+eos+':0') if self.add_eos_token else ''}"
        pair = f"{single}{(' '+bos+':1') if self.add_bos_token else ''} $B:1{(' '+eos+':1') if self.add_eos_token else ''}"

        special_tokens = []
        if self.add_bos_token:
            special_tokens.append((bos, bos_token_id))
        if self.add_eos_token:
            special_tokens.append((eos, eos_token_id))
        self._tokenizer.post_processor = processors.TemplateProcessing(
            single=single, pair=pair, special_tokens=special_tokens
        )

    @property
    def add_eos_token(self):
        return self._add_eos_token

    @property
    def add_bos_token(self):
        return self._add_bos_token

    @add_eos_token.setter
    def add_eos_token(self, value):
        self._add_eos_token = value
        self.update_post_processor()

    @add_bos_token.setter
    def add_bos_token(self, value):
        self._add_bos_token = value
        self.update_post_processor()

    @property
    def default_chat_template(self):
        """
        Cohere Tokenizer uses <|START_OF_TURN_TOKEN|> and <|END_OF_TURN_TOKEN|> to indicate each turn in a chat.
        Additioanlly, to indicate the source of the message, <|USER_TOKEN|>, <|CHATBOT_TOKEN|> and <|SYSTEM_TOKEN|>
        for user, assitant and system messages respectively.

        The output should look something like:
        <|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{ preamble }}<|END_OF_TURN_TOKEN|>
        <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{ How are you? }}<|END_OF_TURN_TOKEN|>
        <|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{{ I am doing well! }}<|END_OF_TURN_TOKEN|>

        Use add_generation_prompt to add a prompt for the model to generate a response:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("CohereForAI/c4ai-command-r-v01", trust_remote_code=True)
        >>> messages = [{"role": "user", "content": "Hello, how are you?"}]
        >>> tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Hello, how are you?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>

        """
        logger.warning_once(
            "\nNo chat template is defined for this tokenizer - using the default template "
            f"for the {self.__class__.__name__} class. If the default is not appropriate for "
            "your model, please set `tokenizer.chat_template` to an appropriate template. "
            "See https://huggingface.co/docs/transformers/main/chat_templating for more information.\n"
        )
        template = (
            "{{ bos_token }}"
            "{% if messages[0]['role'] == 'system' %}"
            "{% set loop_messages = messages[1:] %}"  # Extract system message if it's present
            "{% set system_message = messages[0]['content'] %}"
            "{% elif USE_DEFAULT_PROMPT == true %}"
            "{% set loop_messages = messages %}"  # Or use the default system message if the flag is set
            "{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}"
            "{% else %}"
            "{% set loop_messages = messages %}"
            "{% set system_message = false %}"
            "{% endif %}"
            "{% if system_message != false %}"  # Start with system message
            "{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + system_message + '<|END_OF_TURN_TOKEN|>' }}"
            "{% endif %}"
            "{% for message in loop_messages %}"  # Loop over all non-system messages
            "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
            "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
            "{% endif %}"
            "{% set content = message['content'] %}"
            "{% if message['role'] == 'user' %}"  # After all of that, handle messages/roles in a fairly normal way
            "{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}"
            "{% endif %}"
        )
        template = template.replace("USE_DEFAULT_PROMPT", "true" if self.use_default_system_prompt else "false")
        default_message = DEFAULT_SYSTEM_PROMPT.replace("\n", "\\n").replace("'", "\\'")
        template = template.replace("DEFAULT_SYSTEM_MESSAGE", default_message)

        return template

    @property
    def default_tool_use_template(self):
        template = (
            "{{ bos_token }}"
            "{% if messages[0]['role'] == 'system' %}"
            "{% set loop_messages = messages[1:] %}"  # Extract system message if it's present
            "{% set system_message = messages[0]['content'] %}"
            "{% else %}"
            "{% set loop_messages = messages %}"
            "{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}"
            "{% endif %}"
            "{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}"
            "{{ '# Safety Preamble' }}"
            "{{ '\nThe instructions in this section override those in the task description and style guide sections. Don\\'t answer questions that are harmful or immoral.' }}"
            "{{ '\n\n# System Preamble' }}"
            "{{ '\n## Basic Rules' }}"
            "{{ '\nYou are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user\\'s requests, you cite your sources in your answers, according to those instructions.' }}"
            "{{ '\n\n# User Preamble' }}"
            "{{ '\n' + system_message }}"
            "{{'\n\n## Available Tools\nHere is a list of tools that you have available to you:\n\n'}}"
            "{% for tool in tools %}"
            "{% if loop.index0 != 0 %}"
            "{{ '\n\n'}}"
            "{% endif %}"
            "{{'```python\ndef ' + tool.name + '('}}"
            "{% for param_name, param_fields in tool.parameter_definitions.items() %}"
            "{% if loop.index0 != 0 %}"
            "{{ ', '}}"
            "{% endif %}"
            "{{param_name}}: "
            "{% if not param_fields.required %}"
            "{{'Optional[' + param_fields.type + '] = None'}}"
            "{% else %}"
            "{{ param_fields.type }}"
            "{% endif %}"
            "{% endfor %}"
            "{{ ') -> List[Dict]:\n    \"\"\"'}}"
            "{{ tool.description }}"
            "{% if tool.parameter_definitions|length != 0 %}"
            "{{ '\n\n    Args:\n        '}}"
            "{% for param_name, param_fields in tool.parameter_definitions.items() %}"
            "{% if loop.index0 != 0 %}"
            "{{ '\n        ' }}"
            "{% endif %}"
            "{{ param_name + ' ('}}"
            "{% if not param_fields.required %}"
            "{{'Optional[' + param_fields.type + ']'}}"
            "{% else %}"
            "{{ param_fields.type }}"
            "{% endif %}"
            "{{ '): ' + param_fields.description }}"
            "{% endfor %}"
            "{% endif %}"
            "{{ '\n    \"\"\"\n    pass\n```' }}"
            "{% endfor %}"
            "{{ '<|END_OF_TURN_TOKEN|>'}}"
            "{% for message in loop_messages %}"
            "{% set content = message['content'] %}"
            "{% if message['role'] == 'user' %}"
            "{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}"
            "{% elif message['role'] == 'system' %}"
            "{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}"
            "{% endif %}"
            "{% endfor %}"
            "{{'<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Write \\'Action:\\' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user\\'s last input. You can use any of the supplied tools any number of times, but you should aim to execute the minimum number of necessary actions for the input. You should use the `directly-answer` tool if calling the other tools is unnecessary. The list of actions you want to call should be formatted as a list of json objects, for example:\n```json\n[\n    {\n        \"tool_name\": title of the tool in the specification,\n        \"parameters\": a dict of parameters to input into the tool as they are defined in the specs, or {} if it takes no parameters\n    }\n]```<|END_OF_TURN_TOKEN|>'}}"
            "{% if add_generation_prompt %}"
            "{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}"
            "{% endif %}"
        )
        default_message = DEFAULT_RAG_PREAMBLE.replace("\n", "\\n").replace("'", "\\'")
        template = template.replace("DEFAULT_SYSTEM_MESSAGE", default_message)
        return template

    @property
    def default_grounded_generation_template(self):
        template = (
            "{{ bos_token }}"
            "{% if messages[0]['role'] == 'system' %}"
            "{% set loop_messages = messages[1:] %}"  # Extract system message if it's present
            "{% set system_message = messages[0]['content'] %}"
            "{% else %}"
            "{% set loop_messages = messages %}"
            "{% set system_message = 'DEFAULT_SYSTEM_MESSAGE' %}"
            "{% endif %}"
            "{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}"
            "{{ '# Safety Preamble' }}"
            "{{ '\nThe instructions in this section override those in the task description and style guide sections. Don\\'t answer questions that are harmful or immoral.' }}"
            "{{ '\n\n# System Preamble' }}"
            "{{ '\n## Basic Rules' }}"
            "{{ '\nYou are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user\\'s requests, you cite your sources in your answers, according to those instructions.' }}"
            "{{ '\n\n# User Preamble' }}"
            "{{ '\n' + system_message }}"
            "{{ '<|END_OF_TURN_TOKEN|>'}}"
            "{% for message in loop_messages %}"  # Loop over all non-system messages
            "{% set content = message['content'] %}"
            "{% if message['role'] == 'user' %}"  # After all of that, handle messages/roles in a fairly normal way
            "{{ '<|START_OF_TURN_TOKEN|><|USER_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}"
            "{% elif message['role'] == 'system' %}"
            "{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' + content.strip() + '<|END_OF_TURN_TOKEN|>' }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'  + content.strip() + '<|END_OF_TURN_TOKEN|>' }}"
            "{% endif %}"
            "{% endfor %}"
            "{{ '<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>'}}"
            "{{ '<results>' }}"
            "{% for document in documents %}"  # Loop over all non-system messages
            "{{ '\nDocument: ' }}"
            "{{ loop.index0 }}\n"
            "{% for key, value in document.items() %}"
            "{{ key }}: {{value}}\n"
            "{% endfor %}"
            "{% endfor %}"
            "{{ '</results>'}}"
            "{{ '<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>' }}"
            "{{ 'Carefully perform the following instructions, in order, starting each with a new line.\n' }}"
            "{{ 'Firstly, Decide which of the retrieved documents are relevant to the user\\'s last input by writing \\'Relevant Documents:\\' followed by comma-separated list of document numbers. If none are relevant, you should instead write \\'None\\'.\n' }}"
            "{{ 'Secondly, Decide which of the retrieved documents contain facts that should be cited in a good answer to the user\\'s last input by writing \\'Cited Documents:\\' followed a comma-separated list of document numbers. If you dont want to cite any of them, you should instead write \\'None\\'.\n' }}"
            "{% if citation_mode=='accurate' %}"
            "{{ 'Thirdly, Write \\'Answer:\\' followed by a response to the user\\'s last input in high quality natural english. Use the retrieved documents to help you. Do not insert any citations or grounding markup.\n' }}"
            "{% endif %}"
            "{{ 'Finally, Write \\'Grounded answer:\\' followed by a response to the user\\'s last input in high quality natural english. Use the symbols <co: doc> and </co: doc> to indicate when a fact comes from a document in the search result, e.g <co: 0>my fact</co: 0> for a fact from document 0.' }}"
            "{{ '<|END_OF_TURN_TOKEN|>' }}"
            "{% if add_generation_prompt %}"
            "{{ '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>' }}"
            "{% endif %}"
        )
        default_message = DEFAULT_RAG_PREAMBLE.replace("\n", "\\n").replace("'", "\\'")
        template = template.replace("DEFAULT_SYSTEM_MESSAGE", default_message)
        return template

    def _apply_template_with_arguments(
        self,
        conversation: Union[List[Dict[str, str]], "Conversation"],
        template: Optional[str] = None,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_dict: bool = False,
        **kwargs,
    ) -> Union[str, List[int]]:
        """Just tokenization_utils_base.apply_chat_template, but modified so that the jinjia template can take kwargs"""
        if hasattr(conversation, "messages"):
            # Indicates it's a Conversation object
            conversation = conversation.messages

        # Compilation function uses a cache to avoid recompiling the same template
        compiled_template = self._compile_jinja_template(template)

        rendered = compiled_template.render(
            messages=conversation, add_generation_prompt=add_generation_prompt, **kwargs, **self.special_tokens_map
        )

        if padding is True:
            padding = "max_length"  # There's only one sequence here, so "longest" makes no sense
        if tokenize:
            if return_dict:
                return self(
                    rendered,
                    padding=padding,
                    truncation=truncation,
                    max_length=max_length,
                    add_special_tokens=False,
                    return_tensors=return_tensors,
                    **kwargs,
                )
            else:
                return self.encode(
                    rendered,
                    padding=padding,
                    truncation=truncation,
                    max_length=max_length,
                    add_special_tokens=False,
                    return_tensors=return_tensors,
                    **kwargs,
                )
        else:
            return rendered

    def apply_tool_use_template(
        self,
        conversation: Union[List[Dict[str, str]], "Conversation"],
        tools: List[Dict],
        tool_use_template: Optional[str] = None,
        **kwargs,
    ) -> Union[str, List[int]]:
        """Create a Command-R tool-use prompt.

        Once rendered, the prompt instructs the model to generate a list of actions to perform on a set of user supplied tools
        to help carry out the user's requests.

        Conceptually, this works in the same way as `apply_chat_format`, but takes an additional `tools` parameter.

        Converts a Conversation object or a list of dictionaries with `"role"` and `"content"` keys and a list of available
        tools for the model to use into a prompt string, or a list of token ids.
        This method will use the tokenizer's `default_tool_use_template` template specified at the class level.
        You can override the default template using the `tool_use_template` kwarg but the quality of your results may decrease.

        Args:
            conversation (Union[List[Dict[str, str]], "Conversation"]): A Conversation object or list of dicts
                with "role" and "content" keys, representing the chat history so far.
            tools (List[Dict]): a list of tools to render into the prompt for the model to choose from.
                See an example at the bottom of the docstring.
                The format should be:
                   * name (str): The name of the tool to be called. Valid names contain only the characters a-z,
                        A-Z, 0-9, _ and must not begin with a digit.
                   * description (str): The description of what the tool does, the model uses the description to
                        choose when and how to call the function.
                   * parameter_definitions (List[Dict]): The input parameters of the tool. Accepts a dictionary
                        where the key is the name of the parameter and the value is the parameter spec.
                        Valid parameter names contain only the characters a-z, A-Z, 0-9, _ and must not begin with a digit.
                        Parameter specs are as follows:
                       * description (str): The description of the parameter.
                       * type (str): the type of the parameter - most effective for python builtin data types, such as 'str', 'bool'
                       * required: boolean: Denotes whether the parameter is always present (required) or not. Defaults to not required.
            tool_use_template (str, *optional*): A Jinja template to use for this conversion. If
                this is not passed, the model's default chat template will be used instead.
            add_generation_prompt (bool, *optional*): Whether to end the prompt with the token(s) that indicate
                the start of an assistant message. This is useful when you want to generate a response from the model.
                Note that this argument will be passed to the chat template, and so it must be supported in the
                template for this argument to have any effect.
            tokenize (`bool`, defaults to `True`):
                Whether to tokenize the output. If `False`, the output will be a string.
            padding (`bool`, defaults to `False`):
                Whether to pad sequences to the maximum length. Has no effect if tokenize is `False`.
            truncation (`bool`, defaults to `False`):
                Whether to truncate sequences at the maximum length. Has no effect if tokenize is `False`.
            max_length (`int`, *optional*):
                Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is `False`. If
                not specified, the tokenizer's `max_length` attribute will be used as a default.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Has no effect if tokenize is `False`. Acceptable
                values are:
                - `'tf'`: Return TensorFlow `tf.Tensor` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether to return a dictionary with named outputs. Has no effect if tokenize is `False`.
            **tokenizer_kwargs: Additional kwargs to pass to the tokenizer.

        Returns:
            `str`: A rendered prompt string.
            or if tokenize=True:
            `List[int]`: A list of token ids representing the tokenized chat so far, including control tokens. This
            output is ready to pass to the model, either directly or via methods like `generate()`.

        Examples:

        ```python
        >>> tokenizer = CohereTokenizerFast.from_pretrained("CohereForAI/c4ai-command-r-0.1")
        >>> tools = [
            {
                "name": "internet_search",
                "description": "Returns a list of relevant document snippets for a textual query retrieved from the internet",
                "parameter_definitions": {
                    "query": {
                        "description": "Query to search the internet with",
                        "type": "str",
                        "required": True
                    }
                }
            },
            {
                "name': "directly_answer",
                "description": "Calls a standard (un-augmented) AI chatbot to generate a response given the conversation history",
                "parameter_definitions": {}
            }
        ]
        >>> conversation = [
            {"role": "user", "content": "Whats the biggest penguin in the world?"}
        ]
        >>> # render the prompt, ready for user to inspect, or for input into the model:
        >>> prompt = tokenizer.apply_tool_use_template(
            conversation,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        >>> print(prompt)
        <BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># Safety Preamble
        The instructions in this section override those in the task description and style guide sections. Don't answer questions that are harmful or immoral.

        # System Preamble
        ## Basic Rules
        You are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user's requests, you cite your sources in your answers, according to those instructions.

        # User Preamble
        ## Task and Context
        You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.

        ## Style Guide
        Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.

        ## Available Tools
        Here is a list of tools that you have available to you:

        \\`\\`\\`python
        def internet_search(query: str) -> List[Dict]:
            \"\"\"Returns a list of relevant document snippets for a textual query retrieved from the internet

            Args:
                query (str): Query to search the internet with
            \"\"\"
            pass
        \\`\\`\\`

        \\`\\`\\`python
        def directly_answer() -> List[Dict]:
            \"\"\"Calls a standard (un-augmented) AI chatbot to generate a response given the conversation history
            \"\"\"
            pass
        \\`\\`\\`<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Whats the biggest penguin in the world?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Write 'Action:' followed by a json-formatted list of actions that you want to perform in order to produce a good response to the user's last input. You can use any of the supplied tools any number of times, but you should aim to execute the minimum number of necessary actions for the input. You should use the `directly-answer` tool if calling the other tools is unnecessary. The list of actions you want to call should be formatted as a list of json objects, for example:
        \\`\\`\\`json
        [
            {
                "tool_name": title of the tool in the specification,
                "parameters": a dict of parameters to input into the tool as they are defined in the specs, or {} if it takes no parameters
            }
        ]\\`\\`\\`<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>
        ```
        >>> inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')
        >>> outputs = model.generate(inputs, max_new_tokens=128)
        >>> print(tokenizer.decode(outputs[0]))
        Action: ```json
        [
            {
                "tool_name": "internet_search",
                "parameters": {
                    "query": "biggest penguin in the world"
                }
            }
        ]
        ```
        """
        # priority: `tool_use_template` argument > `tokenizer.tool_use_template` > `tokenizer.default_tool_use_template`
        if tool_use_template is None:
            if self.tool_use_template is not None:
                tool_use_template = self.tool_use_template
            else:
                tool_use_template = self.default_tool_use_template

        return self._apply_template_with_arguments(
            conversation,
            tools=tools,
            template=tool_use_template,
            **kwargs,
        )

    def apply_grounded_generation_template(
        self,
        conversation: Union[List[Dict[str, str]], "Conversation"],
        documents: List[Dict],
        citation_mode: Literal["fast", "accurate"] = "accurate",
        grounded_generation_template: Optional[str] = None,
        **kwargs,
    ) -> Union[str, List[int]]:
        """Create a Command-R grounded generation (aka RAG) prompt.

        Once rendered, the prompt instructs the model to generate a response with citations in, based on supplied documents.

        Conceptually, this works in the same way as `apply_chat_format`, but takes additional `documents`
        and parameter `citation_mode` parameters.

        Converts a Conversation object or a list of dictionaries with `"role"` and `"content"` keys and a list of
        documents for the model to ground its response on into a prompt string, or a list of token ids.
        This method will use the tokenizer's `grounded_generation_template` template specified at the class level.
        You can override the default template using the `grounded_generation_template` kwarg but the quality of your results may decrease.

        Args:
            conversation (Union[List[Dict[str, str]], "Conversation"]): A Conversation object or list of dicts
                with "role" and "content" keys, representing the chat history so far.
            documents (List[Dict[str, str]): A list of dicts, representing documents or tool outputs to ground your
                generation on. A document is a semistructured dict, wiht a string to string mapping. Common fields are
                `url`, `title`, `snippet` etc but should be descriptive of the key. They will get rendered into the prompt.
            citation_mode: either "accurate" (prompt the model to generate an answer first, then rewrite it with citation
                spans in) or "fast", where the prompt instructs the model to generate an answer with citations in directly.
                The former has higher quality citations, the latter requires fewer tokens to be generated.
            grounded_generation_template (str, *optional*): A Jinja template to use for this conversion. If
                this is not passed, the model's default grounded_generation_template template will be used instead.
            add_generation_prompt (bool, *optional*): Whether to end the prompt with the token(s) that indicate
                the start of an assistant message. This is useful when you want to generate a response from the model.
                Note that this argument will be passed to the chat template, and so it must be supported in the
                template for this argument to have any effect.
            tokenize (`bool`, defaults to `True`):
                Whether to tokenize the output. If `False`, the output will be a string.
            padding (`bool`, defaults to `False`):
                Whether to pad sequences to the maximum length. Has no effect if tokenize is `False`.
            truncation (`bool`, defaults to `False`):
                Whether to truncate sequences at the maximum length. Has no effect if tokenize is `False`.
            max_length (`int`, *optional*):
                Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is `False`. If
                not specified, the tokenizer's `max_length` attribute will be used as a default.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Has no effect if tokenize is `False`. Acceptable
                values are:
                - `'tf'`: Return TensorFlow `tf.Tensor` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether to return a dictionary with named outputs. Has no effect if tokenize is `False`.
            **tokenizer_kwargs: Additional kwargs to pass to the tokenizer.

        Returns:
            `str`: A rendered prompt string.
            or if tokenize=True:
            `List[int]`: A list of token ids representing the tokenized chat so far, including control tokens. This
            output is ready to pass to the model, either directly or via methods like `generate()`.

        Examples:

        ```python
        >>> tokenizer = CohereTokenizerFast.from_pretrained('CohereForAI/c4ai-command-r-0.1')

        >>> # define documents:
        >>> documents = [
            { "title": "Tall penguins", "text": "Emperor penguins are the tallest." },
            { "title": "Penguin habitats", "text": "Emperor penguins only live in Antarctica."}
        ]
        >>> # define a conversation:
        >>> conversation = [
            {"role": "user", "content": "Whats the biggest penguin in the world?"}
        ]
        >>> # render the prompt, ready for user to inspect, or for input into the model:
        >>> grounded_generation_prompt = tokenizer.apply_grounded_generation_template(
            conversation,
            documents=documents,
            tokenize=False,
            add_generation_prompt=True,
        )
        >>> print(grounded_generation_prompt)
        <BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># Safety Preamble
        The instructions in this section override those in the task description and style guide sections. Don't answer questions that are harmful or immoral.

        ## Basic Rules
        You are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user's requests, you cite your sources in your answers, according to those instructions.

        # User Preamble
        ## Task and Context
        You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.

        ## Style Guide
        Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>Whats the biggest penguin in the world?<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><results>
        Document: 0
        title: Tall penguins
        text: Emperor penguins are the tallest.

        Document: 1
        title: Penguin habitats
        text: Emperor penguins only live in Antarctica.
        </results><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Carefully perform the following instructions, in order, starting each with a new line.
        Firstly, Decide which of the retrieved documents are relevant to the user's last input by writing 'Relevant Documents:' followed by comma-separated list of document numbers. If none are relevant, you should instead write 'None'.
        Secondly, Decide which of the retrieved documents contain facts that should be cited in a good answer to the user's last input by writing 'Cited Documents:' followed a comma-separated list of document numbers. If you dont want to cite any of them, you should instead write 'None'.
        Thirdly, Write 'Answer:' followed by a response to the user's last input in high quality natural english. Use the retrieved documents to help you. Do not insert any citations or grounding markup.
        Finally, Write 'Grounded answer:' followed by a response to the user's last input in high quality natural english. Use the symbols <co: doc> and </co: doc> to indicate when a fact comes from a document in the search result, e.g <co: 0>my fact</co: 0> for a fact from document 0.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>'''
        ```
        >>> inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')
        >>> outputs = model.generate(inputs, max_new_tokens=128)
        >>> print(tokenizer.decode(outputs[0]))
        Relevant Documents: 0,1
        Cited Documents: 0,1
        Answer: The Emperor Penguin is the tallest or biggest penguin in the world. It is a bird that lives only in Antarctica and grows to a height of around 122 centimetres.
        Grounded answer: The <co: 0>Emperor Penguin</co: 0> is the <co: 0>tallest</co: 0> or biggest penguin in the world. It is a bird that <co: 1>lives only in Antarctica</co: 1> and <co: 0>grows to a height of around 122 centimetres.</co: 0>
        """
        # priority: `grounded_generation_template` argument > `tokenizer.grounded_generation_template` > `tokenizer.default_grounded_generation_template`
        if grounded_generation_template is None:
            if self.grounded_generation_template is not None:
                grounded_generation_template = self.grounded_generation_template
            else:
                grounded_generation_template = self.default_grounded_generation_template

        return self._apply_template_with_arguments(
            conversation,
            documents=documents,
            template=grounded_generation_template,
            citation_mode=citation_mode,
            **kwargs,
        )

    # TODO ArthurZ let's rely on the template processor instead, refactor all fast tokenizers
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output
