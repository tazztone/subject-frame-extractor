"""
Tests for deduplication filtering functionality.

Uses fixtures from conftest.py for mock setup.
"""

from unittest.mock import MagicMock, patch

import imagehash
import numpy as np
import pytest

from core.filtering import _apply_deduplication_filter, _run_batched_lpips


@pytest.fixture
def sample_frames_for_dedup():
    """Sample frames with phash values for deduplication testing."""

    def make_hash(val, size=8):
        arr = np.zeros((size, size), dtype=bool)
        if val == 1:
            arr.fill(True)
        return str(imagehash.ImageHash(arr))

    h1 = make_hash(0)  # All False
    h3 = make_hash(1)  # All True

    # close to h3
    arr4 = np.ones((8, 8), dtype=bool)
    arr4[0, 0] = False  # 1 bit difference
    h4 = str(imagehash.ImageHash(arr4))

    return [
        {"filename": "f1.jpg", "phash": h1, "metrics": {"quality_score": 10}},
        {"filename": "f2.jpg", "phash": h1, "metrics": {"quality_score": 20}},  # better duplicate of f1
        {"filename": "f3.jpg", "phash": h3, "metrics": {"quality_score": 10}},
        {"filename": "f4.jpg", "phash": h4, "metrics": {"quality_score": 5}},  # worse duplicate of f3
    ]


def test_dedup_phash_replacement(sample_frames_for_dedup, mock_thumbnail_manager, mock_config):
    filters = {"enable_dedup": True, "dedup_method": "pHash", "dedup_thresh": 5}

    mask, reasons = _apply_deduplication_filter(
        sample_frames_for_dedup, filters, mock_thumbnail_manager, mock_config, "/tmp"
    )

    # f1 vs f2: f2 is better (20 > 10). f1 should be rejected.
    # f3 vs f4: f3 is better (10 > 5). f4 should be rejected.
    assert not mask[0], f"f1 should be rejected (f2 is better). Reasons: {reasons.get('f1.jpg')}"
    assert mask[1], "f2 should be kept"
    assert mask[2], "f3 should be kept"
    assert not mask[3], "f4 should be rejected (f3 is better)"

    assert "duplicate" in reasons["f1.jpg"]
    assert "duplicate" in reasons["f4.jpg"]


def test_dedup_phash_no_replacement(sample_frames_for_dedup, mock_thumbnail_manager, mock_config):
    # Modify data so duplicates are worse
    sample_frames_for_dedup[1]["metrics"]["quality_score"] = 5  # f2 worse than f1

    filters = {"enable_dedup": True, "dedup_method": "pHash", "dedup_thresh": 5}

    mask, reasons = _apply_deduplication_filter(
        sample_frames_for_dedup, filters, mock_thumbnail_manager, mock_config, "/tmp"
    )

    # f1 vs f2: f1 is better (10 > 5). f2 rejected.
    assert mask[0], "f1 should be kept"
    assert not mask[1], "f2 should be rejected"
    assert mask[2], "f3 should be kept"
    assert not mask[3], "f4 should be rejected"


def test_dedup_disabled(sample_frames_for_dedup, mock_thumbnail_manager, mock_config):
    filters = {"enable_dedup": False}
    mask, reasons = _apply_deduplication_filter(
        sample_frames_for_dedup, filters, mock_thumbnail_manager, mock_config, "/tmp"
    )
    assert np.all(mask)
    assert not reasons


def test_dedup_threshold(sample_frames_for_dedup, mock_thumbnail_manager, mock_config):
    # Set threshold to 0 (exact match only)
    filters = {"enable_dedup": True, "dedup_method": "pHash", "dedup_thresh": 0}

    # f4 (1 bit diff) should NOT be rejected against f3
    mask, reasons = _apply_deduplication_filter(
        sample_frames_for_dedup, filters, mock_thumbnail_manager, mock_config, "/tmp"
    )

    assert not mask[0]  # f1 and f2 are exact duplicates, f2 is better
    assert mask[1]
    assert mask[2]
    assert mask[3], "f4 should be kept because thresh is 0 and it has distance 1"


def test_run_batched_lpips(mock_thumbnail_manager):
    # Setup mocks
    mock_tm = mock_thumbnail_manager
    mock_tm.get.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

    mock_loss_fn = MagicMock()

    mock_tensor = MagicMock()
    mock_tensor.ndim = 1
    mock_tensor.squeeze.return_value = mock_tensor

    mock_cpu_tensor = MagicMock()
    mock_cpu_tensor.numpy.return_value = np.array([0.05, 0.2])
    mock_tensor.cpu.return_value = mock_cpu_tensor

    mock_loss_fn.forward.return_value = mock_tensor

    with patch("core.filtering.get_lpips_metric", return_value=mock_loss_fn):
        all_frames = [
            {"filename": "f1.jpg", "metrics": {"quality_score": 10}},
            {"filename": "f2.jpg", "metrics": {"quality_score": 20}},
            {"filename": "f3.jpg", "metrics": {"quality_score": 10}},
            {"filename": "f4.jpg", "metrics": {"quality_score": 5}},
        ]
        pairs = [(0, 1), (2, 3)]
        dedup_mask = np.array([True, True, True, True])
        reasons = MagicMock()
        reasons.__getitem__.return_value = []

        _run_batched_lpips(pairs, all_frames, dedup_mask, reasons, mock_tm, "/tmp", threshold=0.1)

        # Pair 0 (f1, f2): dist 0.05 <= 0.1. f2 (20) > f1 (10). f1 rejected.
        assert not dedup_mask[0]
        assert dedup_mask[1]

        # Pair 1 (f3, f4): dist 0.2 > 0.1. No rejection.
        assert dedup_mask[2]
        assert dedup_mask[3]
