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

import os
import re
from argparse import ArgumentParser, Namespace
from datetime import date
from pathlib import Path

from ..utils import logging
from . import BaseTransformersCLICommand


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


CURRENT_YEAR = date.today().year
TRANSFORMERS_PATH = Path(__file__).parent.parent
REPO_PATH = TRANSFORMERS_PATH.parent.parent

DEFAULT_CLASS_DOCSTRING = """r\"\"\"
    Constructs a fast {model_name} image processor.

    Args:
        do_resize (`bool`, *optional*):
            Whether to resize the image's (height, width) dimensions to the specified `size`. Can be overridden by the
            `do_resize` parameter in the `preprocess` method.
        size (`dict`, *optional*):
            Size of the output image after resizing. Can be overridden by the `size` parameter in the `preprocess`
            method.
        resample (`PILImageResampling`, *optional*):
            Resampling filter to use if resizing the image. Only has an effect if `do_resize` is set to `True`. Can be
            overridden by the `resample` parameter in the `preprocess` method.
        do_center_crop (`bool`, *optional*, defaults to `True`):
            Whether to center crop the image to the specified `crop_size`. Can be overridden by `do_center_crop` in the
            `preprocess` method.
        crop_size (`Dict[str, int]` *optional*, defaults to 224):
            Size of the output image after applying `center_crop`. Can be overridden by `crop_size` in the `preprocess`
            method.
        do_rescale (`bool`, *optional*):
            Whether to rescale the image by the specified scale `rescale_factor`. Can be overridden by the
            `do_rescale` parameter in the `preprocess` method.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image. Only has an effect if `do_rescale` is set to `True`. Can be
            overridden by the `rescale_factor` parameter in the `preprocess` method.
        do_normalize (`bool`, *optional*):
            Whether to normalize the image. Can be overridden by the `do_normalize` parameter in the `preprocess`
            method. Can be overridden by the `do_normalize` parameter in the `preprocess` method.
        image_mean (`float` or `List[float]`, *optional*):
            Mean to use if normalizing the image. This is a float or list of floats the length of the number of
            channels in the image. Can be overridden by the `image_mean` parameter in the `preprocess` method. Can be
            overridden by the `image_mean` parameter in the `preprocess` method.
        image_std (`float` or `List[float]`, *optional*):
            Standard deviation to use if normalizing the image. This is a float or list of floats the length of the
            number of channels in the image. Can be overridden by the `image_std` parameter in the `preprocess` method.
            Can be overridden by the `image_std` parameter in the `preprocess` method.
        do_convert_rgb (`bool`, *optional*):
            Whether to convert the image to RGB.
    \"\"\"
"""


def add_import_structure_entry_init(content: str, fast_image_processor_name: str, model_name: str):
    """
    Add an entry to the `_import_structure` dictionary in the `__init__.py` file of the transformers package.
    """
    # Step 1: Find the block
    block_regex = re.compile(
        r"if not is_torchvision_available\(\):.*?else:\s*(\n(?P<indent>\s+)_import_structure\[.*?\].*?\n(?:\s*(?P=indent)_import_structure\[.*?\].*?\n)*)",
        re.DOTALL,
    )
    match = block_regex.search(content)

    if not match:
        raise ValueError("Couldn't find the '_import_structure' block.")

    # Capture the block content and indentation
    block_content = match.group(1)
    indent = match.group("indent")

    # Step 2: Parse existing entries
    lines = block_content.strip().split("\n")
    entries = []

    import_structure_header = indent + lines[0]
    entries = lines[1:]

    # Add the new entry, maintaining alphabetical order
    new_entry = f'{indent}_import_structure["models.{model_name}"].append("{fast_image_processor_name}")'
    if new_entry not in entries:
        entries.append(new_entry)

    entries.sort()
    entries = [import_structure_header] + entries

    # Step 3: Reconstruct the block
    updated_block = "\n".join(entry for entry in entries)

    # Replace the original block in the content
    updated_content = content[: match.start(1)] + "\n" + updated_block + "\n" + content[match.end(1) :]

    return updated_content


def add_import_statement_init(content: str, fast_image_processor_name: str, model_name: str):
    """
    Add an import statement to the `__init__.py` file of the transformers package.
    """
    # Step 1: Find the block
    block_regex = re.compile(
        r"if not is_torchvision_available\(\):\s+raise OptionalDependencyNotAvailable\(\)\s+except OptionalDependencyNotAvailable:\s+from \.utils\.dummy_torchvision_objects import \*\s+else:(?P<else_block>\s*(\n\s*from .+ import .*\n)+)(?=\s*# Modeling)",
        re.DOTALL,
    )
    match = block_regex.search(content)

    if match:
        block_content = match.group("else_block")  # The captured import block
    else:
        print("Couldn't find the import statement block.")

    # Step 2: Parse existing entries
    lines = block_content.strip().split("\n")
    entries = []

    indent = " " * (len(lines[1]) - len(lines[1].lstrip()))
    import_structure_header = indent + lines[0]
    entries = lines[1:]

    # Add the new entry, maintaining alphabetical order
    new_entry = f"{indent}from .models.{model_name} import {fast_image_processor_name}"
    if new_entry not in entries:
        entries.append(new_entry)

    entries.sort()
    entries = [import_structure_header] + entries

    # Step 3: Reconstruct the block
    updated_block = "\n".join(entry for entry in entries)

    # Replace the original block in the content
    updated_content = (
        content[: match.start("else_block")] + "\n" + updated_block + "\n\n" + content[match.end("else_block") :]
    )

    return updated_content


def add_fast_image_processor_to_main_init(fast_image_processor_name: str, model_name: str):
    """
    Add the fast image processor to the main __init__.py file of the transformers package.
    """
    with open(TRANSFORMERS_PATH / "__init__.py", "r", encoding="utf-8") as f:
        content = f.read()

    # add _import_structure entry
    content = add_import_structure_entry_init(content, fast_image_processor_name, model_name)
    # add import statement
    content = add_import_statement_init(content, fast_image_processor_name, model_name)

    # write the updated content
    with open(TRANSFORMERS_PATH / "__init__.py", "w", encoding="utf-8") as f:
        f.write(content)


def add_fast_image_processor_to_model_init(
    fast_image_processing_module_file: str, fast_image_processor_name, model_name: str
):
    """
    Add the fast image processor to the __init__.py file of the model.
    """
    with open(TRANSFORMERS_PATH / "models" / model_name / "__init__.py", "r", encoding="utf-8") as f:
        content = f.read()

    fast_image_processing_module_file = fast_image_processing_module_file.split(os.sep)[-1].replace(".py", "")

    if "import *" in content:
        # we have an init file in the updated format
        # get the indented block after if TYPE_CHECKING: and before else:, append the new import, sort the imports and write the updated content
        # Step 1: Find the block
        block_regex = re.compile(
            r"if TYPE_CHECKING:\n(?P<if_block>.*?)(?=\s*else:)",
            re.DOTALL,
        )
        match = block_regex.search(content)

        if not match:
            raise ValueError("Couldn't find the 'if TYPE_CHECKING' block.")

        block_content = match.group("if_block")  # The captured import block

        # Step 2: Parse existing entries
        entries = block_content.split("\n")
        indent = " " * (len(entries[0]) - len(entries[0].lstrip()))
        new_entry = f"{indent}from .{fast_image_processing_module_file} import *"
        if new_entry not in entries:
            entries.append(new_entry)
        entries.sort()
        updated_block = "\n".join(entry for entry in entries)

        # Replace the original block in the content
        updated_content = content[: match.start("if_block")] + updated_block + content[match.end("if_block") :]
    else:
        # we have an init file in the old format

        # add "is_torchvision_available" import to from ...utils import (
        # Regex to match import statements from transformers.utils
        pattern = r"""
            from\s+\.\.\.utils\s+import\s+
            (?:                                   # Non-capturing group for either:
                ([\w, ]+)                         # 1. Single-line imports (e.g., 'a, b')
                |                                 # OR
                \((.*?)\)                         # 2. Multi-line imports (e.g., '(a, ... b)')
            )
        """
        regex = re.compile(pattern, re.VERBOSE | re.DOTALL)

        def replacement_function(match):
            # Extract existing imports
            imports = (match.group(1) or match.group(2)).split(",")
            imports = imports[:-1] if imports[-1] == "\n" else imports
            imports = [imp.strip() for imp in imports]

            # Add the new import if not already present
            if "is_torchvision_available" not in imports:
                imports.append("is_torchvision_available")
                imports.sort()

            # Convert to multi-line import in all cases
            updated_imports = "(\n    " + ",\n    ".join(imports) + ",\n)"

            return f"from ...utils import {updated_imports}"

        # Replace all matches in the file content
        updated_content = regex.sub(replacement_function, content)

        vision_import_structure_block = f'    _import_structure["{fast_image_processing_module_file[:-5]}"] = ["{fast_image_processor_name[:-4]}"]\n'

        added_import_structure_block = (
            "try:\n    if not is_torchvision_available():\n"
            "        raise OptionalDependencyNotAvailable()\n"
            "except OptionalDependencyNotAvailable:\n"
            "    pass\n"
            "else:\n"
            f'    _import_structure["{fast_image_processing_module_file}"] = ["{fast_image_processor_name}"]\n'
        )

        if vision_import_structure_block not in updated_content:
            raise ValueError("Couldn't find the 'vision _import_structure block' block.")

        if added_import_structure_block not in updated_content:
            updated_content = updated_content.replace(
                vision_import_structure_block, vision_import_structure_block + "\n" + added_import_structure_block
            )

        vision_import_statement_block = (
            f"        from .{fast_image_processing_module_file[:-5]} import {fast_image_processor_name[:-4]}\n"
        )

        added_import_statement_block = (
            "    try:\n        if not is_torchvision_available():\n"
            "            raise OptionalDependencyNotAvailable()\n"
            "    except OptionalDependencyNotAvailable:\n"
            "        pass\n"
            "    else:\n"
            f"        from .{fast_image_processing_module_file} import {fast_image_processor_name}\n"
        )

        if vision_import_statement_block not in updated_content:
            raise ValueError("Couldn't find the 'vision _import_structure block' block.")

        if added_import_statement_block not in updated_content:
            updated_content = updated_content.replace(
                vision_import_statement_block, vision_import_statement_block + "\n" + added_import_statement_block
            )

    # write the updated content
    with open(TRANSFORMERS_PATH / "models" / model_name / "__init__.py", "w", encoding="utf-8") as f:
        f.write(updated_content)


def add_fast_image_processor_to_auto(image_processor_name: str, fast_image_processor_name: str):
    """
    Add the fast image processor to the auto module.
    """
    with open(TRANSFORMERS_PATH / "models" / "auto" / "image_processing_auto.py", "r", encoding="utf-8") as f:
        content = f.read()

    # get all lines containing the image processor name
    updated_content = content.replace(
        f'("{image_processor_name}",)', f'("{image_processor_name}", "{fast_image_processor_name}")'
    )

    # write the updated content
    with open(TRANSFORMERS_PATH / "models" / "auto" / "image_processing_auto.py", "w", encoding="utf-8") as f:
        f.write(updated_content)


def add_fast_image_processor_to_dummy(fast_image_processor_name: str):
    """
    Add the fast image processor to the dummy torchvision objects file.
    """
    dummy_torchvision_objects_file = TRANSFORMERS_PATH / "utils" / "dummy_torchvision_objects.py"
    with open(dummy_torchvision_objects_file, "r", encoding="utf-8") as f:
        content = f.read()

    # regex to find objects starting with "class " and ending with "ImageProcessorFast", including "ImageProcessorFast" in the match
    image_processor_names = re.findall(r"class (\w*ImageProcessorFast)", content)
    image_processor_names.append(fast_image_processor_name)
    image_processor_names.sort()
    index_new = image_processor_names.index(fast_image_processor_name)

    new_dummy_object = (
        f"class {fast_image_processor_name}(metaclass=DummyObject):\n"
        '    _backends = ["torchvision"]\n\n'
        "    def __init__(self, *args, **kwargs):\n"
        '        requires_backends(self, ["torchvision"])\n'
    )
    if new_dummy_object not in content:
        if index_new != len(image_processor_names) - 1:
            # add the dummy object just before the next ImageProcessorFast
            first_line = f"class {image_processor_names[index_new+1]}(metaclass=DummyObject):"
            updated_content = content.replace(first_line, new_dummy_object + "\n\n" + first_line)
        else:
            # add the dummy object at the very end
            updated_content = content + "\n\n" + new_dummy_object

        # write the updated content
        with open(dummy_torchvision_objects_file, "w", encoding="utf-8") as f:
            f.write(updated_content)


def add_fast_image_processor_to_doc(fast_image_processor_name: str, model_name: str):
    """
    Add the fast image processor to the model's doc file.
    """
    doc_source = REPO_PATH / "docs" / "source"
    # find the doc files
    doc_files = list(doc_source.glob(f"*/model_doc/{model_name}.md"))
    if not doc_files:
        # try again with "-"
        doc_files = list(doc_source.glob(f"*/model_doc/{model_name.replace('_', '-')}.md"))
    if not doc_files:
        raise ValueError(f"No doc files found for {model_name}")

    base_doc_string = (
        f"## {fast_image_processor_name[:-4]}\n\n" f"[[autodoc]] {fast_image_processor_name[:-4]}\n" "    - preprocess"
    )
    fast_doc_string = (
        f"## {fast_image_processor_name}\n\n" f"[[autodoc]] {fast_image_processor_name}\n" "    - preprocess"
    )

    for doc_file in doc_files:
        with open(doc_file, "r", encoding="utf-8") as f:
            content = f.read()

        if fast_doc_string not in content:
            # add the fast image processor to the doc
            updated_content = content.replace(
                base_doc_string,
                base_doc_string + "\n\n" + fast_doc_string,
            )

            # write the updated content
            with open(doc_file, "w", encoding="utf-8") as f:
                f.write(updated_content)


def add_fast_image_processor_to_tests(fast_image_processor_name: str, model_name: str):
    """
    Add the fast image processor to the image processing tests.
    """
    tests_path = REPO_PATH / "tests" / "models" / model_name
    test_file = tests_path / f"test_image_processing_{model_name}.py"
    if not os.path.exists(test_file):
        logger.warning(f"No test file found for {model_name}. Skipping.")
        return

    with open(test_file, "r", encoding="utf-8") as f:
        content = f.read()

    # add is_torchvision_available import to the imports
    # Regex to match import statements from transformers.utils
    pattern = r"""
        from\s+transformers\.utils\s+import\s+
        (?:                                   # Non-capturing group for either:
            ([\w, ]+)                         # 1. Single-line imports (e.g., 'a, b')
            |                                 # OR
            \((.*?)\)                         # 2. Multi-line imports (e.g., '(a, ... b)')
        )
    """
    regex = re.compile(pattern, re.VERBOSE | re.DOTALL)

    def replacement_function(match):
        # Extract existing imports
        existing_imports = (match.group(1) or match.group(2)).split(",")
        existing_imports = existing_imports[:-1] if existing_imports[-1] == "\n" else existing_imports
        existing_imports = [imp.strip() for imp in existing_imports]

        # Add the new import if not already present
        if "is_torchvision_available" not in existing_imports:
            existing_imports.append("is_torchvision_available")
            existing_imports.sort()

        # Rebuild the import statement
        if match.group(1):  # Single-line import
            updated_imports = ", ".join(existing_imports)
        else:  # Multi-line import
            updated_imports = "(\n    " + ",\n    ".join(existing_imports) + ",\n)"

        return f"from transformers.utils import {updated_imports}"

    # Replace all matches in the file content
    updated_content = regex.sub(replacement_function, content)

    # add the fast image processor to the imports
    base_import_string = f"    from transformers import {fast_image_processor_name[:-4]}"
    fast_import_string = (
        "    if is_torchvision_available():\n" f"        from transformers import {fast_image_processor_name}"
    )
    if fast_import_string not in updated_content:
        updated_content = updated_content.replace(base_import_string, base_import_string + "\n\n" + fast_import_string)

    # get line starting with "    image_processing_class = " and add a line after it starting with "    fast_image_processing_class = "
    image_processing_class_line = re.search(r"    image_processing_class = .*", updated_content)
    if not image_processing_class_line:
        logger.warning(f"Couldn't find the 'image_processing_class' line in {test_file}. Skipping.")
        return

    fast_image_processing_class_line = (
        f"    fast_image_processing_class = {fast_image_processor_name} if is_torchvision_available() else None"
    )
    if "    fast_image_processing_class = " not in updated_content:
        updated_content = updated_content.replace(
            image_processing_class_line.group(0),
            image_processing_class_line.group(0) + "\n" + fast_image_processing_class_line,
        )

    # write the updated content
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(updated_content)


def get_fast_image_processing_content_header(content: str) -> str:
    """
    Get the header of the slow image processor file.
    """
    # get all lines before and including the line containing """Image processor
    content_header = re.search(r"^(.*?\n)*?\"\"\"Image processor.*", content)
    content_header = content_header.group(0)
    content_header = re.sub(r"# Copyright (\d+)\s", f"# Copyright {CURRENT_YEAR} ", content_header)
    content_header = content_header.replace("Image processor", "Fast Image processor")
    return content_header


def write_default_fast_image_processor_file(
    fast_image_processing_module_file: str, fast_image_processor_name: str, content_base_file: str
):
    """
    Write a default fast image processor file. Used when encountering a problem while parsing the slow image processor file.
    """
    imports = "\n\nfrom ...image_processing_utils_fast import BaseImageProcessorFast\n\n\n"
    content_header = get_fast_image_processing_content_header(content_base_file)
    content_base_file = (
        f"class {fast_image_processor_name}(BaseImageProcessorFast):\n"
        "    # To be implemented\n"
        "    resample = None\n"
        "    image_mean = None\n"
        "    image_std = None\n"
        "    size = None\n"
        "    default_to_square = None\n"
        "    crop_size = None\n"
        "    do_resize = None\n"
        "    do_center_crop = None\n"
        "    do_rescale = None\n"
        "    do_normalize = None\n"
        "    do_convert_rgb = None\n\n\n"
        f'__all__ = ["{fast_image_processor_name}"]\n'
    )

    content = content_header + imports + content_base_file

    with open(fast_image_processing_module_file, "w", encoding="utf-8") as f:
        f.write(content)


def add_fast_image_processor_file(
    fast_image_processing_module_file: str, fast_image_processor_name: str, content_base_file: str
):
    """
    Add the fast image processor file to the model's folder.
    """
    # if the file already exists, do nothing
    if os.path.exists(fast_image_processing_module_file):
        print(f"{fast_image_processing_module_file} already exists. Skipping.")
        return

    regex = rf"class {fast_image_processor_name[:-4]}.*?(\n\S|$)"
    match = re.search(regex, content_base_file, re.DOTALL)
    if not match:
        print(f"Couldn't find the {fast_image_processor_name[:-4]} class in {fast_image_processing_module_file}")
        print("Creating a new file with the default content.")
        return write_default_fast_image_processor_file(
            fast_image_processing_module_file, fast_image_processor_name, content_base_file
        )
    # Exclude the last unindented line
    slow_class_content = match.group(0).rstrip()
    # get default args:
    # find the __init__ block which start with def __init__ and ends with def
    match = re.search(r"def __init__.*?def ", slow_class_content, re.DOTALL)
    if not match:
        print(
            f"Couldn't find the __init__ block for {fast_image_processor_name[:-4]} in {fast_image_processing_module_file}"
        )
        print("Creating a new file with the default content.")
        return write_default_fast_image_processor_file(
            fast_image_processing_module_file, fast_image_processor_name, content_base_file
        )
    init = match.group(0)
    init_signature_block = init.split(")")[0]
    arg_names = init_signature_block.split(":")
    arg_names = [arg_name.split("\n")[-1].strip() for arg_name in arg_names]
    # get the default values
    default_args = re.findall(r"= (.*?)(?:,|\))", init_signature_block)

    # build default args dict
    default_args_dict = dict(zip(arg_names, default_args))
    pattern_default_size = r"size = size if size is not None else\s+(.*)"
    match_default_size = re.findall(pattern_default_size, init)
    default_args_dict["size"] = match_default_size[0] if match_default_size else None
    pattern_default_crop_size = r"crop_size = crop_size if crop_size is not None else\s+(.*)"
    match_default_crop_size = re.findall(pattern_default_crop_size, init)
    default_args_dict["crop_size"] = match_default_crop_size[0] if match_default_crop_size else None
    pattern_default_image_mean = r"self.image_mean = image_mean if image_mean is not None else\s+(.*)"
    match_default_image_mean = re.findall(pattern_default_image_mean, init)
    default_args_dict["image_mean"] = match_default_image_mean[0] if match_default_image_mean else None
    pattern_default_image_std = r"self.image_std = image_std if image_std is not None else\s+(.*)"
    match_default_image_std = re.findall(pattern_default_image_std, init)
    default_args_dict["image_std"] = match_default_image_std[0] if match_default_image_std else None
    default_args_dict["default_to_square"] = False if "(size, default_to_square=False" in init else None

    content_header = get_fast_image_processing_content_header(content_base_file)
    class_docstring = DEFAULT_CLASS_DOCSTRING.format(
        model_name=fast_image_processor_name.replace("ImageProcessorFast", "")
    )
    content_base_file = (
        f"class {fast_image_processor_name}(BaseImageProcessorFast):\n"
        f"    {class_docstring}\n\n"
        "    # This generated class can be used as a starting point for the fast image processor.\n"
        "    # if the image processor is only used for simple augmentations, such as resizing, center cropping, rescaling, or normalizing,\n"
        "    # only the default values should be set in the class.\n"
        "    # If the image processor requires more complex augmentations, methods from BaseImageProcessorFast can be overridden.\n"
        "    # For an example of a fast image processor requiring more complex augmentations, see `LlavaOnevisionImageProcessorFast`.\n\n"
        "    # Default values should be checked against the slow image processor\n"
        "    # None values left after checking can be removed\n"
        f'    resample = {default_args_dict.get("resample")}\n'
        f'    image_mean = {default_args_dict.get("image_mean")}\n'
        f'    image_std = {default_args_dict.get("image_std")}\n'
        f'    size = {default_args_dict.get("size")}\n'
        f'    default_to_square = {default_args_dict.get("default_to_square")}\n'
        f'    crop_size = {default_args_dict.get("crop_size")}\n'
        f'    do_resize = {default_args_dict.get("do_resize")}\n'
        f'    do_center_crop = {default_args_dict.get("do_center_crop")}\n'
        f'    do_rescale = {default_args_dict.get("do_rescale")}\n'
        f'    do_normalize = {default_args_dict.get("do_normalize")}\n'
        f'    do_convert_rgb = {default_args_dict.get("do_convert_rgb")}\n\n\n'
        f'__all__ = ["{fast_image_processor_name}"]\n'
    )

    imports = "\n\nfrom ...image_processing_utils_fast import BaseImageProcessorFast\n"
    image_utils_imports = []
    if default_args_dict.get("resample") is not None and "PILImageResampling" in default_args_dict.get("resample"):
        image_utils_imports.append("PILImageResampling")
    if default_args_dict.get("image_mean") is not None and not any(
        char.isdigit() for char in default_args_dict.get("image_mean")
    ):
        image_utils_imports.append(default_args_dict.get("image_mean"))
    if default_args_dict.get("image_std") is not None and not any(
        char.isdigit() for char in default_args_dict.get("image_std")
    ):
        image_utils_imports.append(default_args_dict.get("image_std"))

    if image_utils_imports:
        # sort imports
        image_utils_imports.sort()
        imports += f"from ...image_utils import {', '.join(image_utils_imports)}\n"

    content = content_header + imports + "\n\n" + content_base_file

    with open(fast_image_processing_module_file, "w", encoding="utf-8") as f:
        f.write(content)


def add_fast_image_processor(model_name: str):
    """
    Add the necessary references to the fast image processor in the transformers package,
    and create the fast image processor file in the model's folder.
    """
    model_module = TRANSFORMERS_PATH / "models" / model_name
    image_processing_module_file = list(model_module.glob("image_processing*.py"))
    if not image_processing_module_file:
        raise ValueError(f"No image processing module found in {model_module}")
    elif len(image_processing_module_file) > 1:
        for file_name in image_processing_module_file:
            if not str(file_name).endswith("_fast.py"):
                image_processing_module_file = str(file_name)
                break
    else:
        image_processing_module_file = str(image_processing_module_file[0])

    with open(image_processing_module_file, "r", encoding="utf-8") as f:
        content_base_file = f.read()

    # regex to find object starting with "class " and ending with "ImageProcessor", including "ImageProcessor" in the match
    image_processor_name = re.findall(r"class (\w*ImageProcessor)", content_base_file)
    if not image_processor_name:
        raise ValueError(f"No ImageProcessor class found in {image_processing_module_file}")
    elif len(image_processor_name) > 1:
        raise ValueError(f"Multiple ImageProcessor classes found in {image_processing_module_file}")

    image_processor_name = image_processor_name[0]
    fast_image_processor_name = image_processor_name + "Fast"
    fast_image_processing_module_file = image_processing_module_file.replace(".py", "_fast.py")

    print(f"Adding {fast_image_processor_name} to {fast_image_processing_module_file}")

    add_fast_image_processor_to_main_init(
        fast_image_processor_name=fast_image_processor_name,
        model_name=model_name,
    )

    add_fast_image_processor_to_model_init(
        fast_image_processing_module_file=fast_image_processing_module_file,
        fast_image_processor_name=fast_image_processor_name,
        model_name=model_name,
    )

    add_fast_image_processor_to_auto(
        image_processor_name=image_processor_name,
        fast_image_processor_name=fast_image_processor_name,
    )

    add_fast_image_processor_to_dummy(fast_image_processor_name=fast_image_processor_name)

    add_fast_image_processor_to_doc(
        fast_image_processor_name=fast_image_processor_name,
        model_name=model_name,
    )

    add_fast_image_processor_to_tests(
        fast_image_processor_name=fast_image_processor_name,
        model_name=model_name,
    )

    add_fast_image_processor_file(
        fast_image_processing_module_file=fast_image_processing_module_file,
        fast_image_processor_name=fast_image_processor_name,
        content_base_file=content_base_file,
    )


def add_new_model_like_command_factory(args: Namespace):
    return AddFastImageProcessorCommand(model_name=args.model_name)


class AddFastImageProcessorCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        add_fast_image_processor_parser = parser.add_parser("add-fast-image-processor")
        add_fast_image_processor_parser.add_argument(
            "--model-name",
            type=str,
            required=True,
            help="The name of the folder containing the model's implementation.",
        )
        add_fast_image_processor_parser.set_defaults(func=add_new_model_like_command_factory)

    def __init__(self, model_name: str, *args):
        self.model_name = model_name

    def run(self):
        add_fast_image_processor(model_name=self.model_name)
