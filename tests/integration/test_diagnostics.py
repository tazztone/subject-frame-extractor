import os
from unittest.mock import MagicMock

import torch


def test_torch_is_not_mock():
    print(f"\nIntegration Mode: {os.environ.get('PYTEST_INTEGRATION_MODE')}")
    print(f"Torch type: {type(torch)}")
    assert not isinstance(torch, MagicMock), "Torch IS A MOCK! Pollution detected."


def test_sam2_config_resolvable():
    import os

    import sam2

    sam2_dir = os.path.dirname(sam2.__file__)
    config_path = os.path.join(sam2_dir, "configs/sam2.1/sam2.1_hiera_t.yaml")
    print(f"Candidate SAM2 config path: {config_path}")
    assert os.path.exists(config_path), f"SAM2 config NOT FOUND at {config_path}"
