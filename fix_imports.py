import os
import re

def fix_imports_in_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    # Regex to find imports like `from app.submodule.module import ...`
    content = re.sub(r'from app\.(core|domain|io|logic|masking|ml|pipelines|ui)\.(.*)', r'from .\2', content)
    # Regex to find imports like `from app.submodule import ...`
    content = re.sub(r'from app\.(core|domain|io|logic|masking|ml|pipelines|ui) import (.*)', r'from . import \2', content)

    with open(file_path, 'w') as f:
        f.write(content)

def main():
    app_dir = 'app'
    for filename in os.listdir(app_dir):
        if filename.endswith('.py'):
            file_path = os.path.join(app_dir, filename)
            fix_imports_in_file(file_path)
            print(f"Processed {file_path}")

if __name__ == '__main__':
    main()