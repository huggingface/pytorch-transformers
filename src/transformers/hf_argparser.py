# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import dataclasses
import json
import os
import sys
import types
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, ArgumentTypeError
from copy import copy
from enum import Enum
from inspect import isclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, NewType, Optional, Tuple, Union, get_type_hints

import yaml


DataClass = NewType("DataClass", Any)
DataClassType = NewType("DataClassType", Any)


# From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


def make_choice_type_function(choices: list) -> Callable[[str], Any]:
    """
    Creates a mapping function from each choices string representation to the actual value. Used to support multiple
    value types for a single argument.

    Args:
        choices (list): List of choices.

    Returns:
        Callable[[str], Any]: Mapping function from string representation to actual value for each choice.
    """
    str_to_choice = {str(choice): choice for choice in choices}
    return lambda arg: str_to_choice.get(arg, arg)


def HfArg(
    *,
    aliases: Union[str, List[str]] = None,
    help: str = None,
    default: Any = dataclasses.MISSING,
    default_factory: Callable[[], Any] = dataclasses.MISSING,
    metadata: dict = None,
    **kwargs,
) -> dataclasses.Field:
    """Argument helper enabling a concise syntax to create dataclass fields for parsing with `HfArgumentParser`.

    Example comparing the use of `HfArg` and `dataclasses.field`:
    ```
    @dataclass
    class Args:
        regular_arg: str = dataclasses.field(default="Huggingface", metadata={"aliases": ["--example", "-e"], "help": "This syntax could be better!"})
        hf_arg: str = HfArg(default="Huggingface", aliases=["--example", "-e"], help="What a nice syntax!")
    ```

    Args:
        aliases (Union[str, List[str]], optional):
            Single string or list of strings of aliases to pass on to argparse, e.g. `aliases=["--example", "-e"]`.
            Defaults to None.
        help (str, optional): Help string to pass on to argparse that can be displayed with --help. Defaults to None.
        default (Any, optional):
            Default value for the argument. If not default or default_factory is specified, the argument is required.
            Defaults to dataclasses.MISSING.
        default_factory (Callable[[], Any], optional):
            The default_factory is a 0-argument function called to initialize a field's value. It is useful to provide
            default values for mutable types, e.g. lists: `default_factory=list`. Mutually exclusive with `default=`.
            Defaults to dataclasses.MISSING.
        metadata (dict, optional): Further metadata to pass on to `dataclasses.field`. Defaults to None.

    Returns:
        Field: A `dataclasses.Field` with the desired properties.
    """
    if metadata is None:
        # Important, don't use as default param in function signature because dict is mutable and shared across function calls
        metadata = {}
    if aliases is not None:
        metadata["aliases"] = aliases
    if help is not None:
        metadata["help"] = help

    return dataclasses.field(metadata=metadata, default=default, default_factory=default_factory, **kwargs)


class HfArgumentParser(ArgumentParser):
    """
    This subclass of `argparse.ArgumentParser` uses type hints on dataclasses to generate arguments.

    The class is designed to play well with the native argparse. In particular, you can add more (non-dataclass backed)
    arguments to the parser after initialization and you'll get the output back after parsing as an additional
    namespace. Optional: To create sub argument groups use the `_argument_group_name` attribute in the dataclass.

    Args:
        dataclass_types (`DataClassType` or `Iterable[DataClassType]`, *optional*):
            Dataclass type, or list of dataclass types for which we will "fill" instances with the parsed args.
        kwargs (`Dict[str, Any]`, *optional*):
            Passed to `argparse.ArgumentParser()` in the regular way.
    """

    dataclass_types: Iterable[DataClassType]

    def __init__(self, dataclass_types: Optional[Union[DataClassType, Iterable[DataClassType]]] = None, **kwargs):
        # Make sure dataclass_types is an iterable
        if dataclass_types is None:
            dataclass_types = []
        elif not isinstance(dataclass_types, Iterable):
            dataclass_types = [dataclass_types]

        # To make the default appear when using --help
        if "formatter_class" not in kwargs:
            kwargs["formatter_class"] = ArgumentDefaultsHelpFormatter
        super().__init__(**kwargs)
        if dataclasses.is_dataclass(dataclass_types):
            dataclass_types = [dataclass_types]
        self.dataclass_types = list(dataclass_types)
        for dtype in self.dataclass_types:
            self._add_dataclass_arguments(dtype)

    @staticmethod
    def _parse_dataclass_field(parser: ArgumentParser, field: dataclasses.Field):
        # Long-option strings are conventionlly separated by hyphens rather
        # than underscores, e.g., "--long-format" rather than "--long_format".
        # Argparse converts hyphens to underscores so that the destination
        # string is a valid attribute name. Hf_argparser should do the same.
        long_options = [f"--{field.name}"]
        if "_" in field.name:
            long_options.append(f"--{field.name.replace('_', '-')}")

        kwargs = field.metadata.copy()
        # field.metadata is not used at all by Data Classes,
        # it is provided as a third-party extension mechanism.
        if isinstance(field.type, str):
            raise RuntimeError(
                "Unresolved type detected, which should have been done with the help of "
                "`typing.get_type_hints` method by default"
            )

        aliases = kwargs.pop("aliases", [])
        if isinstance(aliases, str):
            aliases = [aliases]

        origin_type = getattr(field.type, "__origin__", field.type)
        if origin_type is Union or (hasattr(types, "UnionType") and isinstance(origin_type, types.UnionType)):
            if str not in field.type.__args__ and (
                len(field.type.__args__) != 2 or type(None) not in field.type.__args__
            ):
                raise ValueError(
                    "Only `Union[X, NoneType]` (i.e., `Optional[X]`) is allowed for `Union` because"
                    " the argument parser only supports one type per argument."
                    f" Problem encountered in field '{field.name}'."
                )
            if type(None) not in field.type.__args__:
                # filter `str` in Union
                field.type = field.type.__args__[0] if field.type.__args__[1] is str else field.type.__args__[1]
                origin_type = getattr(field.type, "__origin__", field.type)
            elif bool not in field.type.__args__:
                # filter `NoneType` in Union (except for `Union[bool, NoneType]`)
                field.type = (
                    field.type.__args__[0] if isinstance(None, field.type.__args__[1]) else field.type.__args__[1]
                )
                origin_type = getattr(field.type, "__origin__", field.type)

        # A variable to store kwargs for a boolean field, if needed
        # so that we can init a `no_*` complement argument (see below)
        bool_kwargs = {}
        if origin_type is Literal or (isinstance(field.type, type) and issubclass(field.type, Enum)):
            if origin_type is Literal:
                kwargs["choices"] = field.type.__args__
            else:
                kwargs["choices"] = [x.value for x in field.type]

            kwargs["type"] = make_choice_type_function(kwargs["choices"])

            if field.default is not dataclasses.MISSING:
                kwargs["default"] = field.default
            else:
                kwargs["required"] = True
        elif field.type is bool or field.type == Optional[bool]:
            # Copy the currect kwargs to use to instantiate a `no_*` complement argument below.
            # We do not initialize it here because the `no_*` alternative must be instantiated after the real argument
            bool_kwargs = copy(kwargs)

            # Hack because type=bool in argparse does not behave as we want.
            kwargs["type"] = string_to_bool
            if field.type is bool or (field.default is not None and field.default is not dataclasses.MISSING):
                # Default value is False if we have no default when of type bool.
                default = False if field.default is dataclasses.MISSING else field.default
                # This is the value that will get picked if we don't include --{field.name} in any way
                kwargs["default"] = default
                # This tells argparse we accept 0 or 1 value after --{field.name}
                kwargs["nargs"] = "?"
                # This is the value that will get picked if we do --{field.name} (without value)
                kwargs["const"] = True
        elif isclass(origin_type) and issubclass(origin_type, list):
            kwargs["type"] = field.type.__args__[0]
            kwargs["nargs"] = "+"
            if field.default_factory is not dataclasses.MISSING:
                kwargs["default"] = field.default_factory()
            elif field.default is dataclasses.MISSING:
                kwargs["required"] = True
        else:
            kwargs["type"] = field.type
            if field.default is not dataclasses.MISSING:
                kwargs["default"] = field.default
            elif field.default_factory is not dataclasses.MISSING:
                kwargs["default"] = field.default_factory()
            else:
                kwargs["required"] = True
        parser.add_argument(*long_options, *aliases, **kwargs)

        # Add a complement `no_*` argument for a boolean field AFTER the initial field has already been added.
        # Order is important for arguments with the same destination!
        # We use a copy of earlier kwargs because the original kwargs have changed a lot before reaching down
        # here and we do not need those changes/additional keys.
        if field.default is True and (field.type is bool or field.type == Optional[bool]):
            bool_kwargs["default"] = False
            parser.add_argument(
                f"--no_{field.name}",
                f"--no-{field.name.replace('_', '-')}",
                action="store_false",
                dest=field.name,
                **bool_kwargs,
            )

    def _add_dataclass_arguments(self, dtype: DataClassType):
        if hasattr(dtype, "_argument_group_name"):
            parser = self.add_argument_group(dtype._argument_group_name)
        else:
            parser = self

        try:
            type_hints: Dict[str, type] = get_type_hints(dtype)
        except NameError:
            raise RuntimeError(
                f"Type resolution failed for {dtype}. Try declaring the class in global scope or "
                "removing line of `from __future__ import annotations` which opts in Postponed "
                "Evaluation of Annotations (PEP 563)"
            )
        except TypeError as ex:
            # Remove this block when we drop Python 3.9 support
            if sys.version_info[:2] < (3, 10) and "unsupported operand type(s) for |" in str(ex):
                python_version = ".".join(map(str, sys.version_info[:3]))
                raise RuntimeError(
                    f"Type resolution failed for {dtype} on Python {python_version}. Try removing "
                    "line of `from __future__ import annotations` which opts in union types as "
                    "`X | Y` (PEP 604) via Postponed Evaluation of Annotations (PEP 563). To "
                    "support Python versions that lower than 3.10, you need to use "
                    "`typing.Union[X, Y]` instead of `X | Y` and `typing.Optional[X]` instead of "
                    "`X | None`."
                ) from ex
            raise

        for field in dataclasses.fields(dtype):
            if not field.init:
                continue
            field.type = type_hints[field.name]
            self._parse_dataclass_field(parser, field)

    def parse_args_into_dataclasses(
        self,
        args=None,
        return_remaining_strings=False,
        look_for_args_file=True,
        args_filename=None,
        args_file_flag=None,
    ) -> Tuple[DataClass, ...]:
        """
        Parse command-line args into instances of the specified dataclass types.

        This relies on argparse's `ArgumentParser.parse_known_args`. See the doc at:
        docs.python.org/3.7/library/argparse.html#argparse.ArgumentParser.parse_args

        Args:
            args:
                List of strings to parse. The default is taken from sys.argv. (same as argparse.ArgumentParser)
            return_remaining_strings:
                If true, also return a list of remaining argument strings.
            look_for_args_file:
                If true, will look for a ".args" file with the same base name as the entry point script for this
                process, and will append its potential content to the command line args.
            args_filename:
                If not None, will uses this file instead of the ".args" file specified in the previous argument.
            args_file_flag:
                If not None, will look for a file in the command-line args specified with this flag. The flag can be
                specified multiple times and precedence is determined by the order (last one wins).

        Returns:
            Tuple consisting of:

                - the dataclass instances in the same order as they were passed to the initializer.abspath
                - if applicable, an additional namespace for more (non-dataclass backed) arguments added to the parser
                  after initialization.
                - The potential list of remaining argument strings. (same as argparse.ArgumentParser.parse_known_args)
        """

        if args_file_flag or args_filename or (look_for_args_file and len(sys.argv)):
            args_files = []

            if args_filename:
                args_files.append(Path(args_filename))
            elif look_for_args_file and len(sys.argv):
                args_files.append(Path(sys.argv[0]).with_suffix(".args"))

            # args files specified via command line flag should overwrite default args files so we add them last
            if args_file_flag:
                # Create special parser just to extract the args_file_flag values
                args_file_parser = ArgumentParser()
                args_file_parser.add_argument(args_file_flag, type=str, action="append")

                # Use only remaining args for further parsing (remove the args_file_flag)
                cfg, args = args_file_parser.parse_known_args(args=args)
                cmd_args_file_paths = vars(cfg).get(args_file_flag.lstrip("-"), None)

                if cmd_args_file_paths:
                    args_files.extend([Path(p) for p in cmd_args_file_paths])

            file_args = []
            for args_file in args_files:
                if args_file.exists():
                    file_args += args_file.read_text().split()

            # in case of duplicate arguments the last one has precedence
            # args specified via the command line should overwrite args from files, so we add them last
            args = file_args + args if args is not None else file_args + sys.argv[1:]
        namespace, remaining_args = self.parse_known_args(args=args)
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: v for k, v in vars(namespace).items() if k in keys}
            for k in keys:
                delattr(namespace, k)
            obj = dtype(**inputs)
            outputs.append(obj)
        if len(namespace.__dict__) > 0:
            # additional namespace.
            outputs.append(namespace)
        if return_remaining_strings:
            return (*outputs, remaining_args)
        else:
            if remaining_args:
                raise ValueError(f"Some specified arguments are not used by the HfArgumentParser: {remaining_args}")

            return (*outputs,)

    def parse_dict(self, args: Dict[str, Any], allow_extra_keys: bool = False) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all, instead uses a dict and populating the dataclass
        types.

        Args:
            args (`dict`):
                dict containing config values
            allow_extra_keys (`bool`, *optional*, defaults to `False`):
                Defaults to False. If False, will raise an exception if the dict contains keys that are not parsed.

        Returns:
            Tuple consisting of:

                - the dataclass instances in the same order as they were passed to the initializer.
        """
        unused_keys = set(args.keys())
        outputs = []
        for dtype in self.dataclass_types:
            keys = {f.name for f in dataclasses.fields(dtype) if f.init}
            inputs = {k: v for k, v in args.items() if k in keys}
            unused_keys.difference_update(inputs.keys())
            obj = dtype(**inputs)
            outputs.append(obj)
        if not allow_extra_keys and unused_keys:
            raise ValueError(f"Some keys are not used by the HfArgumentParser: {sorted(unused_keys)}")
        return tuple(outputs)

    def parse_json_file(
        self, json_file: Union[str, os.PathLike], allow_extra_keys: bool = False
    ) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all, instead loading a json file and populating the
        dataclass types.

        Args:
            json_file (`str` or `os.PathLike`):
                File name of the json file to parse
            allow_extra_keys (`bool`, *optional*, defaults to `False`):
                Defaults to False. If False, will raise an exception if the json file contains keys that are not
                parsed.

        Returns:
            Tuple consisting of:

                - the dataclass instances in the same order as they were passed to the initializer.
        """
        with open(Path(json_file), encoding="utf-8") as open_json_file:
            data = json.loads(open_json_file.read())
        outputs = self.parse_dict(data, allow_extra_keys=allow_extra_keys)
        return tuple(outputs)

    def parse_yaml_file(
        self, yaml_file: Union[str, os.PathLike], allow_extra_keys: bool = False
    ) -> Tuple[DataClass, ...]:
        """
        Alternative helper method that does not use `argparse` at all, instead loading a yaml file and populating the
        dataclass types.

        Args:
            yaml_file (`str` or `os.PathLike`):
                File name of the yaml file to parse
            allow_extra_keys (`bool`, *optional*, defaults to `False`):
                Defaults to False. If False, will raise an exception if the json file contains keys that are not
                parsed.

        Returns:
            Tuple consisting of:

                - the dataclass instances in the same order as they were passed to the initializer.
        """
        outputs = self.parse_dict(yaml.safe_load(Path(yaml_file).read_text()), allow_extra_keys=allow_extra_keys)
        return tuple(outputs)
