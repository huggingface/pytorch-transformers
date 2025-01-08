import argparse
import difflib
import glob
import logging
from io import StringIO

from create_dependency_mapping import find_priority_list

# Console for rich printing
from modular_model_converter import convert_modular_file
from rich.console import Console
from rich.syntax import Syntax


logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)
console = Console()


def process_file(modular_file_path, generated_modeling_content, file_type="modeling_", fix_and_overwrite=False):
    file_name_prefix = file_type.split("*")[0]
    file_name_suffix = file_type.split("*")[-1] if "*" in file_type else ""
    file_path = modular_file_path.replace("modular_", f"{file_name_prefix}_").replace(".py", f"{file_name_suffix}.py")
    # Read the actual modeling file
    with open(file_path, "r") as modeling_file:
        content = modeling_file.read()
    output_buffer = StringIO(generated_modeling_content[file_type][0])
    output_buffer.seek(0)
    output_content = output_buffer.read()
    diff = difflib.unified_diff(
        output_content.splitlines(),
        content.splitlines(),
        fromfile=f"{file_path}_generated",
        tofile=f"{file_path}",
        lineterm="",
    )
    diff_list = list(diff)
    # Check for differences
    if diff_list:
        if fix_and_overwrite:
            with open(file_path, "w") as modeling_file:
                modeling_file.write(generated_modeling_content[file_type][0])
            console.print(f"[bold blue]Overwritten {file_path} with the generated content.[/bold blue]")
        else:
            console.print(f"\n[bold red]Differences found between the generated code and {file_path}:[/bold red]\n")
            diff_text = "\n".join(diff_list)
            syntax = Syntax(diff_text, "diff", theme="ansi_dark", line_numbers=True)
            console.print(syntax)
        return 1
    else:
        console.print(f"[bold green]No differences found for {file_path}.[/bold green]")
        return 0


def compare_files(modular_file_path, fix_and_overwrite=False):
    # Generate the expected modeling content
    generated_modeling_content = convert_modular_file(modular_file_path)
    diff = 0
    for file_type in generated_modeling_content.keys():
        diff += process_file(modular_file_path, generated_modeling_content, file_type, fix_and_overwrite)
    return diff


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare modular_xxx.py files with modeling_xxx.py files.")
    parser.add_argument(
        "--files", default=["all"], type=list, nargs="+", help="List of modular_xxx.py files to compare."
    )
    parser.add_argument(
        "--fix_and_overwrite", action="store_true", help="Overwrite the modeling_xxx.py file if differences are found."
    )
    args = parser.parse_args()
    if args.files == ["all"]:
        args.files = glob.glob("src/transformers/models/**/modular_*.py", recursive=True)
    non_matching_files = 0
    for modular_file_path in find_priority_list(args.files):
        non_matching_files += compare_files(modular_file_path, args.fix_and_overwrite)

    if non_matching_files and not args.fix_and_overwrite:
        raise ValueError("Some diff and their modeling code did not match.")
