import sys
from unittest.mock import patch

from tests.conftest import modules_to_mock

for name in sorted(modules_to_mock.keys()):
    sys.modules[name] = modules_to_mock[name]
import torch

print(f"torch is {torch}")
from core.system_health import check_environment

with (
    patch("torch.cuda.is_available", return_value=True, create=True),
    patch("torch.cuda.get_device_name", return_value="Test GPU", create=True),
    patch("torch.version.cuda", "12.1", create=True),
):
    report = check_environment()
    print("REPORT:")
    for line in report:
        print(line)
