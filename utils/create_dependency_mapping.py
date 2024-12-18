import ast
from collections import defaultdict


# Function to perform topological sorting
def topological_sort(dependencies):
    new_dependencies = {}
    graph = defaultdict(list)
    for node, deps in dependencies.items():
        for dep in deps:
            if "example" not in node and "auto" not in dep:
                graph[dep.split(".")[-2]].append(node.split("/")[-2])
        new_dependencies[node.split("/")[-2]] = node

    # Create a graph and in-degree count for each node
    def filter_one_by_one(filtered_list, reverse):
        if len(reverse) == 0:
            return filtered_list

        graph = defaultdict(list)
        # Build the graph
        for node, deps in reverse.items():
            for dep in deps:
                graph[dep].append(node)

        base_modules = set(reverse.keys()) - set(graph.keys())
        if base_modules == reverse.keys():
            # we are at the end
            return filtered_list + list(graph.keys())
        to_add = []
        for k in graph.keys():
            if len(graph[k]) == 1 and graph[k][0] in base_modules:
                if graph[k][0] in reverse:
                    del reverse[graph[k][0]]
                if k not in filtered_list:
                    to_add += [k]
        for k in base_modules:
            if k not in filtered_list:
                to_add += [k]
        filtered_list += list(to_add)
        return filter_one_by_one(filtered_list, reverse)

    final_order = filter_one_by_one([], graph)

    return [new_dependencies.get(k) for k in final_order if k in new_dependencies]


# Function to extract class and import info from a file
def extract_classes_and_imports(file_path):
    with open(file_path, "r") as file:
        tree = ast.parse(file.read(), filename=file_path)
    imports = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = node.module if isinstance(node, ast.ImportFrom) else None
            if module and (".modeling_" in module):
                imports.add(module)
    return imports


# Function to map dependencies between classes
def map_dependencies(py_files):
    dependencies = defaultdict(set)
    # First pass: Extract all classes and map to files
    for file_path in py_files:
        # dependencies[file_path].add(None)
        class_to_file = extract_classes_and_imports(file_path)
        for module in class_to_file:
            dependencies[file_path].add(module)
    return dependencies


def find_priority_list(py_files):
    dependencies = map_dependencies(py_files)
    ordered_classes = topological_sort(dependencies)
    return ordered_classes
