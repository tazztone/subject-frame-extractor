import os
import sys


def test_debug_env():
    mode = os.environ.get("PYTEST_INTEGRATION_MODE")
    print(f"\nPYTEST_INTEGRATION_MODE: {mode}")
    print(f"insightface in sys.modules: {'insightface' in sys.modules}")
    if "insightface" in sys.modules:
        print(f"insightface type: {type(sys.modules['insightface'])}")
        print(f"insightface module: {sys.modules['insightface']}")

    import torch

    print(f"torch is mock: {hasattr(torch, '__qualname__') == False and 'mock' in str(type(torch))}")
