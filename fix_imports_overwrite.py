import os
import re

def fix_imports_in_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Regex to find imports like `from .module import ...` and replace with `from app.module import ...`
    content = re.sub(r'from \.(.*) import (.*)', r'from app.\1 import \2', content)

    with open(file_path, 'w') as f:
        f.write(content)

def main():
    app_dir = 'app'
    # Exclude files that have already been fixed
    exclude_files = ['composition.py', 'app_ui.py']
    for filename in os.listdir(app_dir):
        if filename.endswith('.py') and filename not in exclude_files:
            file_path = os.path.join(app_dir, filename)
            fix_imports_in_file(file_path)
            print(f"Processed {file_path}")

if __name__ == '__main__':
    main()