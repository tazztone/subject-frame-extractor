from unittest.mock import patch

import pytest

from core.managers.tracker_factory import build_tracker

# Coverage target: Tracker factory selection logic including CUDA availability checks
# Previously uncovered: SAM2.1 selection and missing fallback tests


def test_selects_sam21():
    """Test that requesting SAM2 raises a ValueError indicating it is retired."""
    with pytest.raises(ValueError, match="retired"):
        build_tracker("sam2", "/tmp/model.pt", "cuda")


@patch("core.managers.sam3.SAM3Wrapper")
def test_selects_sam3(mock_sam3):
    """Test that SAM3Wrapper is selected when SAM3 is requested."""
    tracker = build_tracker("sam3", "/tmp/model.pt", "cpu")

    mock_sam3.assert_called_once_with("/tmp/model.pt", "cpu", config=None)
    assert tracker == mock_sam3.return_value


def test_selects_invalid_tracker():
    """Test that an invalid tracker backend raises ValueError."""
    with pytest.raises(ValueError, match="Unknown tracker backend: 'invalid'"):
        build_tracker("invalid", "/tmp/model.pt", "cuda")
