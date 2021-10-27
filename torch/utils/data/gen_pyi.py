import os
from typing import List

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

def find_file_paths(dir_path: str) -> List[str]:
    all_files = os.listdir(dir_path)
    python_files = {fname for fname in all_files if ".py" == fname[-3:]}
    filter_files = {fname for fname in python_files if fname not in files_to_exclude and fname not in deprecated_files}
    paths = {os.path.join(dir_path, fname)for fname in filter_files}
    return paths


def parse_datapipe_file(filename):
    method_to_signature = {}
    with open(filename) as f:
        open_paren_count = 0
        method_name = None
        signature = ""
        skip = False
        for line in f.readlines():
            if line.count("\"\"\"") % 2 == 1:
                skip = not skip
            if skip or "\"\"\"" in line:  # Skipping comment/example blocks
                continue
            if "@functional_datapipe" in line:
                if "(\"" in line:
                    start = line.find("(\"") + len("(\"")
                    end = line.find("\")")
                    method_name = line[start:end]
                elif "(\'" in line:
                    start = line.find("(\'") + len("(\'")
                    end = line.find("\')")
                    method_name = line[start:end]
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
                    method_to_signature[method_name] = signature
                    method_name = None
                    signature = ""
                elif open_paren_count < 0:
                    raise RuntimeError("open parenthesis count < 0. This shouldn't be possible.")
                else:
                    signature += line.strip('\n').strip(' ')

    return method_to_signature

iter_datapipes_file_path = "datapipes/iter"
file_paths = find_file_paths("datapipes/iter")
methods_and_signatures = {}
for file in file_paths:
    methods_and_signatures.update(parse_datapipe_file(file))
for k, v in methods_and_signatures.items():
    print()
    print(f"method: {k}")
    tokens = v.split(',')
    tokens = [t.strip(' ') for t in tokens if t.strip(' ') != ""]
    v = ', '.join(tokens)
    print(v)


