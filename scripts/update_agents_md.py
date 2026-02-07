"""
Generate AGENTS_CODE_REFERENCE.md - Auto-generated code skeleton reference.

Usage: python scripts/update_agents_md.py

This script only generates AGENTS_CODE_REFERENCE.md with code skeletons.
AGENTS.md is now a manually-maintained file and is not touched by this script.
"""

import ast
import datetime
import os
from pathlib import Path



def sanitize_value(node_value):
    """Sanitizes string values in AST nodes to preventing leaking secrets."""
    if isinstance(node_value, ast.Constant) and isinstance(node_value.value, str):
        # Redact all string literals in assignments
        return '"<REDACTED_STRING>"'
    return ast.unparse(node_value)


def process_node(node, indent=0):
    """Convert an AST node to a skeleton representation."""
    lines = []
    prefix = " " * indent

    # Decorators
    for decorator in node.decorator_list:
        lines.append(f"{prefix}@{ast.unparse(decorator)}")

    # Definition line construction
    definition = ""
    if isinstance(node, ast.ClassDef):
        bases = [ast.unparse(b) for b in node.bases]
        base_str = f"({', '.join(bases)})" if bases else ""
        definition = f"{prefix}class {node.name}{base_str}:"
    elif isinstance(node, ast.FunctionDef):
        args = ast.unparse(node.args)
        returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        definition = f"{prefix}def {node.name}({args}){returns}:"
    elif isinstance(node, ast.AsyncFunctionDef):
        args = ast.unparse(node.args)
        returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        definition = f"{prefix}async def {node.name}({args}){returns}:"

    # Collect body content
    body_lines = []

    # Docstring - Compact to single line
    docstring = ast.get_docstring(node)
    if docstring:
        summary = docstring.strip().split("\n")[0]
        if len(summary) > 80:
            summary = summary[:77] + "..."
        body_lines.append(f'{prefix}    """{summary}"""')

    # Child nodes
    has_children = False
    if isinstance(node, ast.ClassDef):
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                has_children = True
                body_lines.extend(process_node(child, indent + 4))
            elif isinstance(child, ast.Assign):
                try:
                    # Limit assignments to one line and 100 chars to save space
                    # REDACTION APPLIED HERE
                    targets = [ast.unparse(t) for t in child.targets]
                    value_str = sanitize_value(child.value)
                    assign_str = f"{' = '.join(targets)} = {value_str}"
                    
                    if len(assign_str) > 80:
                        assign_str = assign_str[:77] + "..."
                    body_lines.append(f"{prefix}    {assign_str}")
                    has_children = True
                except Exception:
                    pass

    # Assemble lines
    if not docstring and not has_children:
        # Condensed single-line format for empty bodies
        lines.append(f"{definition} ...")
    else:
        # Standard format
        lines.append(definition)
        lines.extend(body_lines)

    return lines


def parse_file_to_skeleton(file_path):
    """Parse a Python file and return its skeleton."""
    try:
        code = file_path.read_text(encoding="utf-8")
        tree = ast.parse(code)
    except (SyntaxError, UnicodeDecodeError) as e:
        return f"# Error parsing file: {e}"

    lines = []

    # Docstring - specific to module level
    docstring = ast.get_docstring(tree)
    if docstring:
        summary = docstring.strip().split("\n")[0]
        if len(summary) > 80:
            summary = summary[:77] + "..."
        lines.append(f'"""{summary}"""')

    # Imports - Skipped for compactness

    # Definitions
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            lines.extend(process_node(node))
            # No spacer between top-level definitions for compactness
        elif isinstance(node, ast.Assign):
            # Top level assignments (constants etc)
            try:
                # Limit length
                # REDACTION APPLIED HERE
                targets = [ast.unparse(t) for t in node.targets]
                value_str = sanitize_value(node.value)
                assign_str = f"{' = '.join(targets)} = {value_str}"

                if len(assign_str) > 80:
                    assign_str = assign_str[:77] + "..."
                lines.append(assign_str)
            except Exception:
                pass

    result = "\n".join(lines).strip()
    return result


def generate_file_tree(root_dir):
    """Generate a visual file tree of the project."""
    output = ["## Project Structure\n", "```text"]

    exclude_dirs = {
        ".git",
        "venv",
        ".venv",
        "__pycache__",
        "SAM3_repo",
        ".claude",
        ".jules",
        "dry-run-assets",
        "logs",
        "downloads",
        "models",
        "node_modules",
        ".github",
        "init_logs",
        "ux_reports",
        "test_results",
        ".pytest_cache",
        "__pycache__",
    }

    exclude_files = {
        "AGENTS_CODE_REFERENCE.md",
        ".DS_Store",
        ".gitignore",
        ".gitmodules",
        "test_output.txt",
        "test_results.txt",
        "LICENSE",
        "pyproject.toml",
        ".env_example",
    }

    tree_lines = []

    def walk(directory, prefix=""):
        try:
            items = os.listdir(directory)
        except OSError:
            return
        
        items.sort()

        # Filter items
        filtered_items = []
        for item in items:
            if item in exclude_files:
                continue
            if item.startswith("."):  # Skip dotfiles mostly
                continue
            
            path = os.path.join(directory, item)
            if os.path.isdir(path):
                if item not in exclude_dirs:
                    filtered_items.append(item)
            else:
                # Include relevant source files
                if item.endswith((".py", ".md", ".json", ".txt", ".cfg", ".yml", ".yaml", ".sh")):
                    filtered_items.append(item)

        count = len(filtered_items)
        for i, item in enumerate(filtered_items):
            path = os.path.join(directory, item)
            is_last = i == count - 1
            connector = "└── " if is_last else "├── "

            tree_lines.append(f"{prefix}{connector}{item}")

            if os.path.isdir(path):
                extension = "    " if is_last else "│   "
                walk(path, prefix + extension)

    output.append(".")
    walk(root_dir)
    output.extend(tree_lines)
    output.append("```\n")
    return "\n".join(output)


def generate_skeleton_section(source_dirs):
    """Generate code skeleton section for all Python files in specific source directories."""
    output = ["## Code Skeleton Reference\n"]

    files = []
    
    # Iterate over explicit source directories to be safe (Whitelisting)
    for source_dir in source_dirs:
        root_path = Path(source_dir)
        if not root_path.exists():
            continue
            
        if root_path.is_file():
             if root_path.suffix == ".py":
                 files.append(root_path)
        else:
            for root, dirs, filenames in os.walk(root_path):
                # Exclude hidden and venv dirs even within whitelisted sources
                dirs[:] = [d for d in dirs if not d.startswith(".") and d != "venv" and d != "__pycache__"]
                
                for name in filenames:
                    if name.endswith(".py") and name != os.path.basename(__file__):
                        files.append(Path(root) / name)

    files.sort()

    for file_path in files:
        skeleton = parse_file_to_skeleton(file_path)
        # Skip empty skeletons
        if not skeleton:
            continue

        rel_path = file_path
        output.append(f"### `📄 {rel_path}`\n")
        output.append("```python")
        output.append(skeleton)
        output.append("```\n")

    return "\n".join(output)


def main():
    print("Generating file tree...")
    file_tree = generate_file_tree(".")

    print("Generating code skeletons...")
    # Safe Whitelist of source directories
    source_dirs = ["app.py", "core", "ui", "scripts"]
    main_skeleton = generate_skeleton_section(source_dirs)

    # Tests skeleton
    tests_skeleton = generate_skeleton_section(["tests"])



    # Generate AGENTS_CODE_REFERENCE.md
    header_main = """---
description: Auto-generated code skeletons for the main application.
---

# Code Skeleton Reference

> **⚠️ AUTO-GENERATED FILE**: This file is generated by `scripts/update_agents_md.py`.
> Do not edit manually. Run `python scripts/update_agents_md.py` to regenerate.

This file contains auto-generated code skeletons for the main application.
For test references, see [tests/TESTS_CODE_REFERENCE.md](tests/TESTS_CODE_REFERENCE.md).
For developer guidelines, see [AGENTS.md](AGENTS.md).

{file_tree}
"""

    Path("docs/AGENTS_CODE_REFERENCE.md").write_text(
        header_main.format(file_tree=file_tree) + main_skeleton, encoding="utf-8"
    )
    print("Successfully updated AGENTS_CODE_REFERENCE.md")

    # Generate tests/TESTS_CODE_REFERENCE.md
    header_tests = """---
description: Auto-generated code skeletons for the test suite.
---

# Tests Code Reference

> **⚠️ AUTO-GENERATED FILE**: This file is generated by `scripts/update_agents_md.py`.
> Do not edit manually.

This file contains auto-generated code skeletons for the test suite.
"""

    tests_ref_path = Path("tests/TESTS_CODE_REFERENCE.md")
    tests_ref_path.parent.mkdir(exist_ok=True)
    tests_ref_path.write_text(header_tests + tests_skeleton, encoding="utf-8")
    print(f"Successfully updated {tests_ref_path}")


if __name__ == "__main__":
    main()
