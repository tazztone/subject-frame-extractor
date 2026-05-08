#!/usr/bin/env python3
"""
Scans the codebase for TODO items and generates a prioritized report.
"""

import os
import re
from datetime import datetime
from pathlib import Path


def scan_todos(root_dir="."):
    todo_pattern = re.compile(r"TODO:\s*(.*)", re.IGNORECASE)
    results = []

    # Files to exclude
    exclude_dirs = {".git", ".venv", "__pycache__", "models", "downloads", "outputs", "SAM3_repo"}
    exclude_files = {"generate_todo_report.py"}

    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file in exclude_files:
                continue

            file_path = Path(root) / file
            if file_path.suffix not in {".py", ".md", ".sh", ".yaml", ".yml"}:
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f, 1):
                        match = todo_pattern.search(line)
                        if match:
                            results.append(
                                {
                                    "file": str(file_path),
                                    "line": i,
                                    "text": match.group(1).strip(),
                                    "priority": "High" if "CRITICAL" in line or "IMPORTANT" in line else "Medium",
                                }
                            )
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

    return results


def generate_report(todos):
    report = [
        "# Technical Debt / TODO Report",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total items: {len(todos)}",
        "",
        "## Top Priority Items",
        "",
    ]

    # Sort by priority and then file
    sorted_todos = sorted(todos, key=lambda x: (x["priority"] != "High", x["file"]))

    for todo in sorted_todos:
        report.append(f"- [{todo['priority']}] **{todo['file']}:{todo['line']}**: {todo['text']}")

    return "\n".join(report)


if __name__ == "__main__":
    print("Scanning for TODOs...")
    todos = scan_todos()
    report = generate_report(todos)

    output_path = Path("TODO_REPORT.md")
    output_path.write_text(report)
    print(f"Report generated: {output_path}")
