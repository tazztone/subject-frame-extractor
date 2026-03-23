"""
Unit tests for Tracker Factory.
"""

from unittest.mock import MagicMock, patch

import pytest

from core.managers.tracker_factory import build_tracker


def test_build_tracker_sam2():
    """Verify factory returns SAM21Wrapper for 'sam2' backend."""
    from core.managers.sam21 import SAM21Wrapper

    # We mock SAM21Wrapper directly
    with patch("core.managers.sam21.SAM21Wrapper") as mock_sam2:
        mock_sam2.return_value = MagicMock(spec=SAM21Wrapper)

        tracker = build_tracker("sam2", "dummy.pt", device="cpu")

        assert tracker == mock_sam2.return_value
        mock_sam2.assert_called_once_with("dummy.pt", "cpu")


def test_build_tracker_sam3():
    """Verify factory returns SAM3Wrapper for 'sam3' backend."""
    from core.managers.sam3 import SAM3Wrapper

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


def test_sam3_not_imported_at_module_load():
    """SAM3 must only be imported lazily inside build_tracker, not at module level."""
    import importlib
    import sys

    # Ensure core.managers.tracker_factory doesn't pull in sam3 on import
    if "core.managers.sam3" in sys.modules:
        del sys.modules["core.managers.sam3"]
    # Re-import tracker_factory to check its top-level imports
    importlib.reload(sys.modules["core.managers.tracker_factory"])
    assert "core.managers.sam3" not in sys.modules, "SAM3 must not be imported at tracker_factory module load time"
