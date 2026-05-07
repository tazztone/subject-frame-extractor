import os
import re
from pathlib import Path

def scan_todos(root_dir):
    exclude_dirs = {'.git', 'venv', 'SAM3_repo', '__pycache__', '.claude', '.jules', '.gemini'}
    todo_pattern = re.compile(r'TODO[:\s]+(.*)', re.IGNORECASE)

    todos = []

    for root, dirs, files in os.walk(root_dir):
        # Filter directories in-place
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file.endswith(('.py', '.md', '.sh', '.json', '.txt')):
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for i, line in enumerate(f, 1):
                            match = todo_pattern.search(line)
                            if match:
                                text = match.group(1).strip()
                                todos.append({
                                    'file': str(file_path.relative_to(root_dir)),
                                    'line': i,
                                    'text': text
                                })
                except (UnicodeDecodeError, PermissionError):
                    continue
    return todos

def prioritize_todos(todos):
    # Simple heuristic for "actionability"
    action_keywords = ['implement', 'add', 'fix', 'refactor', 'update', 'remove', 'migrate', 'validate']
    critical_keywords = ['critical', 'bug', 'error', 'crash', 'fail']

    for todo in todos:
        score = 0
        text_lower = todo['text'].lower()

        # Keyword checks
        for kw in action_keywords:
            if kw in text_lower:
                score += 5

        for kw in critical_keywords:
            if kw in text_lower:
                score += 10

        # Length check - longer TODOs might be more descriptive
        score += min(len(todo['text']) // 10, 5)

        # Penalize vague ones
        if len(todo['text']) < 10:
            score -= 10

        todo['score'] = score

    # Sort by score descending
    return sorted(todos, key=lambda x: x['score'], reverse=True)

def main():
    root = Path('.')
    todos = scan_todos(root)
    prioritized = prioritize_todos(todos)

    top_20 = prioritized[:20]

    print("## 📋 TODO Summary")
    print(f"\nTotal TODO items found: {len(todos)}. Below are the top 20 most actionable items:\n")

    for i, todo in enumerate(top_20, 1):
        # Format as a task list item
        print(f"{i}. [ ] **{todo['file']}:{todo['line']}**: {todo['text']}")

if __name__ == "__main__":
    main()
