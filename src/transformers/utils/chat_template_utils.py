# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import inspect
import json
import re
import types
from contextlib import contextmanager
from datetime import datetime
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, get_args, get_origin, get_type_hints

from packaging import version

from .import_utils import is_jinja_available, is_torch_available, is_vision_available


if is_jinja_available():
    import jinja2
    from jinja2.ext import Extension
    from jinja2.sandbox import ImmutableSandboxedEnvironment
else:
    jinja2 = None

if is_vision_available():
    from PIL.Image import Image

if is_torch_available():
    from torch import Tensor


BASIC_TYPES = (int, float, str, bool, Any, type(None), ...)
# Extracts the initial segment of the docstring, containing the function description
description_re = re.compile(r"^(.*?)[\n\s]*(Args:|Returns:|Raises:|\Z)", re.DOTALL)
# Extracts the Args: block from the docstring
args_re = re.compile(r"\n\s*Args:\n\s*(.*?)[\n\s]*(Returns:|Raises:|\Z)", re.DOTALL)
# Splits the Args: block into individual arguments
args_split_re = re.compile(
    r"""
(?:^|\n)  # Match the start of the args block, or a newline
\s*(\w+):\s*  # Capture the argument name and strip spacing
(.*?)\s*  # Capture the argument description, which can span multiple lines, and strip trailing spacing
(?=\n\s*\w+:|\Z)  # Stop when you hit the next argument or the end of the block
""",
    re.DOTALL | re.VERBOSE,
)
# Extracts the Returns: block from the docstring, if present. Note that most chat templates ignore the return type/doc!
returns_re = re.compile(r"\n\s*Returns:\n\s*(.*?)[\n\s]*(Raises:|\Z)", re.DOTALL)


class TypeHintParsingException(Exception):
    """Exception raised for errors in parsing type hints to generate JSON schemas"""

    pass


class DocstringParsingException(Exception):
    """Exception raised for errors in parsing docstrings to generate JSON schemas"""

    pass


def _get_json_schema_type(param_type: str) -> Dict[str, str]:
    type_mapping = {
        int: {"type": "integer"},
        float: {"type": "number"},
        str: {"type": "string"},
        bool: {"type": "boolean"},
        Any: {},
    }
    if is_vision_available():
        type_mapping[Image] = {"type": "image"}
    if is_torch_available():
        type_mapping[Tensor] = {"type": "audio"}
    return type_mapping.get(param_type, {"type": "object"})


def _parse_type_hint(hint: str) -> Dict:
    origin = get_origin(hint)
    args = get_args(hint)

    if origin is None:
        try:
            return _get_json_schema_type(hint)
        except KeyError:
            raise TypeHintParsingException(
                "Couldn't parse this type hint, likely due to a custom class or object: ", hint
            )

    elif origin is Union or (hasattr(types, "UnionType") and origin is types.UnionType):
        # Recurse into each of the subtypes in the Union, except None, which is handled separately at the end
        subtypes = [_parse_type_hint(t) for t in args if t is not type(None)]
        if len(subtypes) == 1:
            # A single non-null type can be expressed directly
            return_dict = subtypes[0]
        elif all(isinstance(subtype["type"], str) for subtype in subtypes):
            # A union of basic types can be expressed as a list in the schema
            return_dict = {"type": sorted([subtype["type"] for subtype in subtypes])}
        else:
            # A union of more complex types requires "anyOf"
            return_dict = {"anyOf": subtypes}
        if type(None) in args:
            return_dict["nullable"] = True
        return return_dict

    elif origin is list:
        if not args:
            return {"type": "array"}
        else:
            # Lists can only have a single type argument, so recurse into it
            return {"type": "array", "items": _parse_type_hint(args[0])}

    elif origin is tuple:
        if not args:
            return {"type": "array"}
        if len(args) == 1:
            raise TypeHintParsingException(
                f"The type hint {str(hint).replace('typing.', '')} is a Tuple with a single element, which "
                "we do not automatically convert to JSON schema as it is rarely necessary. If this input can contain "
                "more than one element, we recommend "
                "using a List[] type instead, or if it really is a single element, remove the Tuple[] wrapper and just "
                "pass the element directly."
            )
        if ... in args:
            raise TypeHintParsingException(
                "Conversion of '...' is not supported in Tuple type hints. "
                "Use List[] types for variable-length"
                " inputs instead."
            )
        return {"type": "array", "prefixItems": [_parse_type_hint(t) for t in args]}

    elif origin is dict:
        # The JSON equivalent to a dict is 'object', which mandates that all keys are strings
        # However, we can specify the type of the dict values with "additionalProperties"
        out = {"type": "object"}
        if len(args) == 2:
            out["additionalProperties"] = _parse_type_hint(args[1])
        return out

    raise TypeHintParsingException("Couldn't parse this type hint, likely due to a custom class or object: ", hint)


def _convert_type_hints_to_json_schema(func: Callable) -> Dict:
    type_hints = get_type_hints(func)
    signature = inspect.signature(func)
    required = []
    for param_name, param in signature.parameters.items():
        if param.annotation == inspect.Parameter.empty:
            raise TypeHintParsingException(f"Argument {param.name} is missing a type hint in function {func.__name__}")
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    properties = {}
    for param_name, param_type in type_hints.items():
        properties[param_name] = _parse_type_hint(param_type)

    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required

    return schema


def parse_google_format_docstring(docstring: str) -> Tuple[Optional[str], Optional[Dict], Optional[str]]:
    """
    Parses a Google-style docstring to extract the function description,
    argument descriptions, and return description.

    Args:
        docstring (str): The docstring to parse.

    Returns:
        The function description, arguments, and return description.
    """

    # Extract the sections
    description_match = description_re.search(docstring)
    args_match = args_re.search(docstring)
    returns_match = returns_re.search(docstring)

    # Clean and store the sections
    description = description_match.group(1).strip() if description_match else None
    docstring_args = args_match.group(1).strip() if args_match else None
    returns = returns_match.group(1).strip() if returns_match else None

    # Parsing the arguments into a dictionary
    if docstring_args is not None:
        docstring_args = "\n".join([line for line in docstring_args.split("\n") if line.strip()])  # Remove blank lines
        matches = args_split_re.findall(docstring_args)
        args_dict = {match[0]: re.sub(r"\s*\n+\s*", " ", match[1].strip()) for match in matches}
    else:
        args_dict = {}

    return description, args_dict, returns


def get_json_schema(func: Callable) -> Dict:
    """
    This function generates a JSON schema for a given function, based on its docstring and type hints. This is
    mostly used for passing lists of tools to a chat template. The JSON schema contains the name and description of
    the function, as well as the names, types and descriptions for each of its arguments. `get_json_schema()` requires
    that the function has a docstring, and that each argument has a description in the docstring, in the standard
    Google docstring format shown below. It also requires that all the function arguments have a valid Python type hint.

    Although it is not required, a `Returns` block can also be added, which will be included in the schema. This is
    optional because most chat templates ignore the return value of the function.

    Args:
        func: The function to generate a JSON schema for.

    Returns:
        A dictionary containing the JSON schema for the function.

    Examples:
    ```python
    >>> def multiply(x: float, y: float):
    >>>    '''
    >>>    A function that multiplies two numbers
    >>>
    >>>    Args:
    >>>        x: The first number to multiply
    >>>        y: The second number to multiply
    >>>    '''
    >>>    return x * y
    >>>
    >>> print(get_json_schema(multiply))
    {
        "name": "multiply",
        "description": "A function that multiplies two numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "x": {"type": "number", "description": "The first number to multiply"},
                "y": {"type": "number", "description": "The second number to multiply"}
            },
            "required": ["x", "y"]
        }
    }
    ```

    The general use for these schemas is that they are used to generate tool descriptions for chat templates that
    support them, like so:

    ```python
    >>> from transformers import AutoTokenizer
    >>> from transformers.utils import get_json_schema
    >>>
    >>> def multiply(x: float, y: float):
    >>>    '''
    >>>    A function that multiplies two numbers
    >>>
    >>>    Args:
    >>>        x: The first number to multiply
    >>>        y: The second number to multiply
    >>>    return x * y
    >>>    '''
    >>>
    >>> multiply_schema = get_json_schema(multiply)
    >>> tokenizer = AutoTokenizer.from_pretrained("CohereForAI/c4ai-command-r-v01")
    >>> messages = [{"role": "user", "content": "What is 179 x 4571?"}]
    >>> formatted_chat = tokenizer.apply_chat_template(
    >>>     messages,
    >>>     tools=[multiply_schema],
    >>>     chat_template="tool_use",
    >>>     return_dict=True,
    >>>     return_tensors="pt",
    >>>     add_generation_prompt=True
    >>> )
    >>> # The formatted chat can now be passed to model.generate()
    ```

    Each argument description can also have an optional `(choices: ...)` block at the end, such as
    `(choices: ["tea", "coffee"])`, which will be parsed into an `enum` field in the schema. Note that this will
    only be parsed correctly if it is at the end of the line:

    ```python
    >>> def drink_beverage(beverage: str):
    >>>    '''
    >>>    A function that drinks a beverage
    >>>
    >>>    Args:
    >>>        beverage: The beverage to drink (choices: ["tea", "coffee"])
    >>>    '''
    >>>    pass
    >>>
    >>> print(get_json_schema(drink_beverage))
    ```
    {
        'name': 'drink_beverage',
        'description': 'A function that drinks a beverage',
        'parameters': {
            'type': 'object',
            'properties': {
                'beverage': {
                    'type': 'string',
                    'enum': ['tea', 'coffee'],
                    'description': 'The beverage to drink'
                    }
                },
            'required': ['beverage']
        }
    }
    """
    doc = inspect.getdoc(func)
    if not doc:
        raise DocstringParsingException(
            f"Cannot generate JSON schema for {func.__name__} because it has no docstring!"
        )
    doc = doc.strip()
    main_doc, param_descriptions, return_doc = parse_google_format_docstring(doc)

    json_schema = _convert_type_hints_to_json_schema(func)
    if (return_dict := json_schema["properties"].pop("return", None)) is not None:
        if return_doc is not None:  # We allow a missing return docstring since most templates ignore it
            return_dict["description"] = return_doc
    for arg, schema in json_schema["properties"].items():
        if arg not in param_descriptions:
            raise DocstringParsingException(
                f"Cannot generate JSON schema for {func.__name__} because the docstring has no description for the argument '{arg}'"
            )
        desc = param_descriptions[arg]
        enum_choices = re.search(r"\(choices:\s*(.*?)\)\s*$", desc, flags=re.IGNORECASE)
        if enum_choices:
            schema["enum"] = [c.strip() for c in json.loads(enum_choices.group(1))]
            desc = enum_choices.string[: enum_choices.start()].strip()
        schema["description"] = desc

    output = {"name": func.__name__, "description": main_doc, "parameters": json_schema}
    if return_dict is not None:
        output["return"] = return_dict
    return {"type": "function", "function": output}


def _render_with_assistant_indices(
    compiled_template, messages, tools, documents, add_generation_prompt, **template_kwargs
):
    rendered_blocks = []
    generation_indices = []
    with compiled_template.environment.activate_tracker(rendered_blocks, generation_indices):
        for block in compiled_template.generate(
            messages=messages,
            tools=tools,
            documents=documents,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        ):
            rendered_blocks.append(block)
        rendered_chat = "".join(rendered_blocks)
    return rendered_chat, generation_indices


@lru_cache
def _compile_jinja_template(chat_template):
    if not is_jinja_available():
        raise ImportError(
            "apply_chat_template requires jinja2 to be installed. Please install it using `pip install jinja2`."
        )

    class AssistantTracker(Extension):
        # This extension is used to track the indices of assistant-generated tokens in the rendered chat
        tags = {"generation"}

        def __init__(self, environment: ImmutableSandboxedEnvironment):
            # The class is only initiated by jinja.
            super().__init__(environment)
            environment.extend(activate_tracker=self.activate_tracker)
            self._rendered_blocks = None
            self._generation_indices = None

        def parse(self, parser: jinja2.parser.Parser) -> jinja2.nodes.CallBlock:
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
            return jinja2.nodes.CallBlock(self.call_method("_generation_support"), [], [], body).set_lineno(lineno)

        @jinja2.pass_eval_context
        def _generation_support(self, context: jinja2.nodes.EvalContext, caller: jinja2.runtime.Macro) -> str:
            rv = caller()
            if self.is_active():
                # Only track generation indices if the tracker is active
                start_index = len("".join(self._rendered_blocks))
                end_index = start_index + len(rv)
                self._generation_indices.append((start_index, end_index))
            return rv

        def is_active(self) -> bool:
            return self._rendered_blocks or self._generation_indices

        @contextmanager
        def activate_tracker(self, rendered_blocks: List[int], generation_indices: List[int]):
            try:
                if self.is_active():
                    raise ValueError("AssistantTracker should not be reused before closed")
                self._rendered_blocks = rendered_blocks
                self._generation_indices = generation_indices

                yield
            finally:
                self._rendered_blocks = None
                self._generation_indices = None

    if version.parse(jinja2.__version__) < version.parse("3.1.0"):
        raise ImportError(
            "apply_chat_template requires jinja2>=3.1.0 to be installed. Your version is " f"{jinja2.__version__}."
        )

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        # We override the built-in tojson filter because Jinja's default filter escapes HTML characters
        # We also expose some options like custom indents and separators
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    def strftime_now(format):
        return datetime.now().strftime(format)

    jinja_env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[AssistantTracker, jinja2.ext.loopcontrols]
    )
    jinja_env.filters["tojson"] = tojson
    jinja_env.globals["raise_exception"] = raise_exception
    jinja_env.globals["strftime_now"] = strftime_now
    return jinja_env.from_string(chat_template)
