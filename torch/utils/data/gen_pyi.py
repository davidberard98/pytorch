import os
from typing import List, Set
from tools.codegen.gen import FileManager

"""
.pyi generation for functional DataPipes Process
# 0. Setup template dataset.pyi.in
# 1. Find files that we want to process (exclude the ones who don't)
# 2. Parse method name and signature
# 3. Remove first argument after self (unless it is "*datapipes")
# 4. Inject file into template dataset.pyi.in
"""


files_to_exclude = {"__init__.py", "utils.py"}
deprecated_files = {"httpreader.py", "linereader.py", "tararchivereader.py", "ziparchivereader.py"}


def find_file_paths(dir_path: str) -> Set[str]:
    all_files = os.listdir(dir_path)
    python_files = {fname for fname in all_files if ".py" == fname[-3:]}
    filter_files = {fname for fname in python_files if fname not in files_to_exclude and fname not in deprecated_files}
    paths = {os.path.join(dir_path, fname)for fname in filter_files}
    return paths


def extract_method_name(line: str) -> str:
    if "(\"" in line:
        start_token, end_token = "(\"", "\")"
    elif "(\'" in line:
        start_token, end_token = "(\'", "\')"
    else:
        raise RuntimeError(f"Unable to find appropriate method name within line:\n{line}")
    start, end = line.find(start_token) + len(start_token), line.find(end_token)
    return line[start:end]


def parse_datapipe_file(filename):
    method_to_signature = {}
    with open(filename) as f:
        open_paren_count = 0
        method_name, signature = "", ""
        skip = False
        for line in f.readlines():
            if line.count("\"\"\"") % 2 == 1:
                skip = not skip
            if skip or "\"\"\"" in line:  # Skipping comment/example blocks
                continue
            if "@functional_datapipe" in line:
                method_name = extract_method_name(line)
                continue
            if method_name and ("def __init__(" in line or "def __new__(" in line):
                open_paren_count += 1
                start = line.find("(") + len("(")
                line = line[start:]
            if open_paren_count > 0:
                open_paren_count += line.count('(')
                open_paren_count -= line.count(')')
                if open_paren_count == 0:
                    end = line.rfind(')')
                    signature += line[:end]
                    method_to_signature[method_name] = process_signature(signature)
                    method_name, signature = "", ""
                elif open_paren_count < 0:
                    raise RuntimeError("open parenthesis count < 0. This shouldn't be possible.")
                else:
                    signature += line.strip('\n').strip(' ')
    return method_to_signature


# Split on comma unless it is within a bracket '[]'
def split_outside_bracket(line: str, delimiter: str = ",") -> List[str]:
    bracket_count = 0
    curr_token = ""
    res = []
    for char in line:
        if char == "[":
            bracket_count += 1
        elif char == "]":
            bracket_count -= 1
        elif char == delimiter and bracket_count == 0:
            res.append(curr_token)
            curr_token = ""
            continue
        curr_token += char
    res.append(curr_token)
    return res


def process_signature(line: str) -> str:
    tokens: List[str] = split_outside_bracket(line)
    for i, token in enumerate(tokens):
        tokens[i] = token.strip(' ')
        if token == "cls":
            tokens[i] = "self"
        elif i > 0 and ("self" == tokens[i - 1]) and (tokens[i][0] != "*"):
            # Remove the datapipe after 'self' or 'cls' unless it has '*'
            tokens[i] = ""
        elif "Callable =" in token:  # Remove default argument if it is a function
            head, default_arg = token.rsplit("=", 2)
            tokens[i] = head.strip(' ') + "= ..."
    tokens = [t for t in tokens if t != ""]
    line = ', '.join(tokens)
    return line


iter_datapipes_file_path = "datapipes/iter"
file_paths = find_file_paths("datapipes/iter")
methods_and_signatures = {}
for file in file_paths:
    methods_and_signatures.update(parse_datapipe_file(file))


method_definitions = []
for method_name, signature in methods_and_signatures.items():
    # TODO: We can add output type here, but they are all
    #     1) mostly IterDataPipe
    #     2) need to fix/double-check typing first within each DataPipe
    # method_definitions.append(f"def {method_name}({signature}) -> {output_type}: ...")
    method_definitions.append(f"def {method_name}({signature}): ...")

fm = FileManager(install_dir='.', template_dir='.', dry_run=False)
fm.write_with_template(filename="dataset.pyi",
                       template_fn="dataset.pyi.in",
                       env_callable=lambda: {'IterableDataPipeMethods': method_definitions})
