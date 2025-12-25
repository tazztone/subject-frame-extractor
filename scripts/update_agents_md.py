"""
Generate AGENTS_CODE_REFERENCE.md - Auto-generated code skeleton reference.

Usage: python scripts/update_agents_md.py

This script only generates AGENTS_CODE_REFERENCE.md with code skeletons.
AGENTS.md is now a manually-maintained file and is not touched by this script.
"""
import ast
import os
from pathlib import Path
import datetime


def process_node(node, indent=0):
    """Convert an AST node to a skeleton representation."""
    lines = []
    prefix = " " * indent

    # Decorators
    for decorator in node.decorator_list:
        lines.append(f"{prefix}@{ast.unparse(decorator)}")

    # Definition
    if isinstance(node, ast.ClassDef):
        bases = [ast.unparse(b) for b in node.bases]
        base_str = f"({', '.join(bases)})" if bases else ""
        lines.append(f"{prefix}class {node.name}{base_str}:")
    elif isinstance(node, ast.FunctionDef):
        args = ast.unparse(node.args)
        returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        lines.append(f"{prefix}def {node.name}({args}){returns}:")
    elif isinstance(node, ast.AsyncFunctionDef):
        args = ast.unparse(node.args)
        returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        lines.append(f"{prefix}async def {node.name}({args}){returns}:")

    # Docstring
    docstring = ast.get_docstring(node)
    if docstring:
        doc_lines = docstring.strip().split('\n')
        lines.append(f'{prefix}    """')
        for dl in doc_lines:
            lines.append(f"{prefix}    {dl.strip()}")
        lines.append(f'{prefix}    """')

    # Body Summary
    if isinstance(node, ast.ClassDef):
        has_content = False
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                has_content = True
                lines.extend(process_node(child, indent + 4))
            elif isinstance(child, ast.Assign):
                try:
                    lines.append(f"{prefix}    {ast.unparse(child)}")
                    has_content = True
                except: pass
        if not has_content:
            lines.append(f"{prefix}    ...")
    else:
        lines.append(f"{prefix}    ...")

    return lines


def parse_file_to_skeleton(file_path):
    """Parse a Python file and return its skeleton."""
    try:
        code = file_path.read_text(encoding='utf-8')
        tree = ast.parse(code)
    except (SyntaxError, UnicodeDecodeError) as e:
        return f"# Error parsing file: {e}"

    lines = []

    # Docstring
    docstring = ast.get_docstring(tree)
    if docstring:
        lines.append(f'"""\n{docstring}\n"""\n')

    # Imports
    imports = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imports.append(ast.unparse(node))

    if imports:
        lines.extend(imports)
        lines.append("")

    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            lines.extend(process_node(node))
            lines.append("")
        elif isinstance(node, ast.Assign):
            lines.append(ast.unparse(node))

    return "\n".join(lines)


def generate_file_tree(root_dir):
    """Generate a visual file tree of the project."""
    output = ["## Project Structure\n", "```text"]

    exclude_dirs = {
        '.git', 'venv', '__pycache__', 'SAM3_repo',
        '.claude', '.jules', 'dry-run-assets', 'logs', 'downloads',
        'models', 'node_modules', '.github', 'init_logs', 'ux_reports',
        'test_results', '.pytest_cache', '__pycache__'
    }

    exclude_files = {
        'AGENTS_CODE_REFERENCE.md', '.DS_Store', '.gitignore', '.gitmodules',
        'test_output.txt', 'test_results.txt', 'LICENSE', 'setup.cfg', '.env_example'
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
            if item.startswith('.'): # Skip dotfiles mostly
                continue

            path = os.path.join(directory, item)
            if os.path.isdir(path):
                if item not in exclude_dirs:
                    filtered_items.append(item)
            else:
                # Include relevant source files
                if item.endswith(('.py', '.md', '.json', '.txt', '.cfg', '.yml', '.yaml', '.sh')):
                     filtered_items.append(item)

        count = len(filtered_items)
        for i, item in enumerate(filtered_items):
            path = os.path.join(directory, item)
            is_last = (i == count - 1)
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


def generate_skeleton_section(root_dir):
    """Generate code skeleton section for all Python files."""
    output = ["## Code Skeleton Reference\n"]

    exclude_dirs = {
        '.git', 'venv', '__pycache__', 'SAM3_repo',
        '.claude', '.jules', 'dry-run-assets', 'logs', 'downloads',
        'models', 'node_modules', '.github', 'init_logs'
    }

    files = []
    for root, dirs, filenames in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for name in filenames:
            if name.endswith(".py") and name != os.path.basename(__file__):
                files.append(Path(root) / name)

    files.sort()

    for file_path in files:
        rel_path = file_path.relative_to(root_dir)
        output.append(f"### `📄 {rel_path}`\n")
        output.append("```python")
        skeleton = parse_file_to_skeleton(file_path)
        output.append(skeleton)
        output.append("```\n")

    return "\n".join(output)


def main():
    current_date = datetime.date.today().isoformat()

    print("Generating file tree...")
    file_tree = generate_file_tree(".")

    print("Generating code skeletons...")
    skeleton_section = generate_skeleton_section(".")

    # Generate AGENTS_CODE_REFERENCE.md
    header = f"""---
Last Updated: {current_date}
---

# Code Skeleton Reference

> **⚠️ AUTO-GENERATED FILE**: This file is generated by `scripts/update_agents_md.py`.
> Do not edit manually. Run `python scripts/update_agents_md.py` to regenerate.

This file contains auto-generated code skeletons for quick reference.
For developer guidelines, see [AGENTS.md](AGENTS.md).

{file_tree}
"""
    
    content = header + skeleton_section
    output_path = Path("AGENTS_CODE_REFERENCE.md")
    output_path.write_text(content, encoding="utf-8")
    print(f"Successfully updated {output_path}")


if __name__ == "__main__":
    main()
