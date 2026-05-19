import pytest
import json
import os
import sys
import ast
from unittest.mock import MagicMock

# Ensure we can import from core and SAM3_repo
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "SAM3_repo"))

# Mock pkg_resources and torch before importing sam3 if needed
sys.modules['pkg_resources'] = MagicMock()
# Torch should be available in the environment but let's be safe if it's not
try:
    import torch
except ImportError:
    sys.modules['torch'] = MagicMock()

def test_coco_loader_patch_applied(tmp_path):
    # 1. Apply patches
    from core.sam3_patches import apply_patches
    apply_patches()

    from sam3.train.data.coco_json_loaders import COCO_FROM_JSON

    # 2. Create dummy COCO file
    d = tmp_path / "data"
    d.mkdir()
    p = d / "dummy_coco.json"
    dummy_coco = {
        "images": [{"id": 1, "width": 100, "height": 100, "file_name": "test.jpg"}],
        "annotations": [],
        "categories": [{"id": 1, "name": "test_cat"}]
    }
    p.write_text(json.dumps(dummy_coco))
    dummy_coco_file = str(p)

    # 3. Test legitimate prompt
    prompts_str = "[{'id': 1, 'name': 'test_cat'}]"
    loader = COCO_FROM_JSON(dummy_coco_file, prompts=prompts_str)
    assert loader.prompts == {1: 'test_cat'}

    # 4. Test malicious prompt
    # Note: If patched, it uses ast.literal_eval which raises ValueError/SyntaxError
    # If NOT patched, it uses eval() which might execute code or fail differently
    malicious_prompt = "[{'id': 1, 'name': 'test'}] + [__import__('os').system('echo VULNERABLE')]"

    # Check if the file exploited.txt is NOT created
    if os.path.exists("exploited_patch.txt"):
        os.remove("exploited_patch.txt")

    malicious_prompt_file = "[{'id': 1, 'name': 'test'}] + [__import__('os').system('echo VULNERABLE > exploited_patch.txt')]"

    try:
        COCO_FROM_JSON(dummy_coco_file, prompts=malicious_prompt_file)
    except (ValueError, SyntaxError, NameError):
        pass

    assert not os.path.exists("exploited_patch.txt"), "Vulnerability still exists! eval() was executed."

if __name__ == "__main__":
    # Manually run if needed
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmpdir:
        test_coco_loader_patch_applied(Path(tmpdir))
        print("Test passed!")
