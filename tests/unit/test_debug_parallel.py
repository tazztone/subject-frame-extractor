import sys

import torch

from core.system_health import check_environment


def test_debug_torch_path():
    print(f"\nWORKER DEBUG: torch.__file__ = {getattr(torch, '__file__', 'MOCKED')}")
    print(f"WORKER DEBUG: torch.cuda.is_available() = {torch.cuda.is_available()}")
    report = check_environment()
    print("WORKER DEBUG: report lines:")
    for line in report:
        print(f"  > {line}")


def test_debug_sys_modules():
    print(f"WORKER DEBUG: 'torch' in sys.modules = {'torch' in sys.modules}")
    print(f"WORKER DEBUG: type(sys.modules['torch']) = {type(sys.modules['torch'])}")
