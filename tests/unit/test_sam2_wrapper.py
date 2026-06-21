import pytest

from core.managers.sam2 import SAM2Wrapper

pytestmark = [pytest.mark.sam2]


def test_sam2_wrapper_init_retired():
    """Verify that SAM2Wrapper immediately raises a ValueError upon instantiation in wrapper tests."""
    with pytest.raises(ValueError, match="retired"):
        SAM2Wrapper(checkpoint_path="dummy.pt")
