"""
Unit tests for Tracker Factory.
"""

from unittest.mock import MagicMock, patch

import pytest

from core.managers.sam3 import SAM3Wrapper
from core.managers.sam21 import SAM21Wrapper
from core.managers.tracker_factory import build_tracker


def test_build_tracker_sam2():
    """Verify factory returns SAM21Wrapper for 'sam2' backend."""
    with patch("core.managers.sam21.SAM21Wrapper") as mock_sam2:
        mock_sam2.return_value = MagicMock(spec=SAM21Wrapper)

        tracker = build_tracker("sam2", "dummy.pt", device="cpu")

        assert tracker == mock_sam2.return_value
        mock_sam2.assert_called_once_with("dummy.pt", "cpu")


def test_build_tracker_sam3():
    """Verify factory returns SAM3Wrapper for 'sam3' backend."""
    # We mock SAM3Wrapper directly because it has heavy imports/Triton logic
    with patch("core.managers.sam3.SAM3Wrapper") as mock_sam3:
        mock_sam3.return_value = MagicMock(spec=SAM3Wrapper)

        tracker = build_tracker("sam3", "dummy.pt", device="cpu")

        assert tracker == mock_sam3.return_value
        mock_sam3.assert_called_once_with("dummy.pt", "cpu")


def test_build_tracker_invalid():
    """Verify factory raises ValueError for invalid backend."""
    with pytest.raises(ValueError) as exc_info:
        build_tracker("invalid_backend", "dummy.pt")

    assert "Unknown tracker backend" in str(exc_info.value)
