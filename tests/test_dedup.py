import pytest
import numpy as np
import imagehash
import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Extensive mocking to survive app.py import
modules_to_mock = {
    'sam3': MagicMock(),
    'sam3.model_builder': MagicMock(),
    'sam3.model.sam3_video_predictor': MagicMock(),
    'mediapipe': MagicMock(),
    'mediapipe.tasks': MagicMock(),
    'mediapipe.tasks.python': MagicMock(),
    'mediapipe.tasks.python.vision': MagicMock(),
    'pyiqa': MagicMock(),
    'scenedetect': MagicMock(),
    'lpips': MagicMock(),
    'yt_dlp': MagicMock(),
    'numba': MagicMock(),
    'matplotlib': MagicMock(),
    'matplotlib.pyplot': MagicMock(),
    'matplotlib.ticker': MagicMock(),
    'torch': MagicMock(),
    'torchvision': MagicMock(),
    'torchvision.ops': MagicMock(),
    'torchvision.transforms': MagicMock(),
    'insightface': MagicMock(),
    'insightface.app': MagicMock(),
}

# Apply mocks
patcher = patch.dict(sys.modules, modules_to_mock)
patcher.start()

from app import _apply_deduplication_filter, Config, ThumbnailManager

@pytest.fixture
def mock_thumbnail_manager():
    return MagicMock(spec=ThumbnailManager)

@pytest.fixture
def sample_frames_for_dedup():
    # Helper to create hash
    def make_hash(val, size=8):
        arr = np.zeros((size, size), dtype=bool)
        if val == 1: arr.fill(True)
        return str(imagehash.ImageHash(arr))

    h1 = make_hash(0) # All False
    h3 = make_hash(1) # All True

    # close to h3
    arr4 = np.ones((8, 8), dtype=bool)
    arr4[0,0] = False # 1 bit difference
    h4 = str(imagehash.ImageHash(arr4))

    return [
        {'filename': 'f1.jpg', 'phash': h1, 'metrics': {'quality_score': 10}},
        {'filename': 'f2.jpg', 'phash': h1, 'metrics': {'quality_score': 20}}, # better duplicate of f1
        {'filename': 'f3.jpg', 'phash': h3, 'metrics': {'quality_score': 10}},
        {'filename': 'f4.jpg', 'phash': h4, 'metrics': {'quality_score': 5}},  # worse duplicate of f3
    ]

def test_dedup_phash_replacement(sample_frames_for_dedup, mock_thumbnail_manager):
    filters = {"enable_dedup": True, "dedup_method": "pHash", "dedup_thresh": 5}
    config = Config()

    mask, reasons = _apply_deduplication_filter(sample_frames_for_dedup, filters, mock_thumbnail_manager, config, "/tmp")

    # f1 vs f2: f2 is better (20 > 10). f1 should be rejected.
    # f3 vs f4: f3 is better (10 > 5). f4 should be rejected.

    assert not mask[0], f"f1 should be rejected (f2 is better). Reasons: {reasons.get('f1.jpg')}"
    assert mask[1], "f2 should be kept"
    assert mask[2], "f3 should be kept"
    assert not mask[3], "f4 should be rejected (f3 is better)"

    assert 'duplicate' in reasons['f1.jpg']
    assert 'duplicate' in reasons['f4.jpg']

def test_dedup_phash_no_replacement(sample_frames_for_dedup, mock_thumbnail_manager):
    # Modify data so duplicates are worse
    sample_frames_for_dedup[1]['metrics']['quality_score'] = 5 # f2 worse than f1

    filters = {"enable_dedup": True, "dedup_method": "pHash", "dedup_thresh": 5}
    config = Config()

    mask, reasons = _apply_deduplication_filter(sample_frames_for_dedup, filters, mock_thumbnail_manager, config, "/tmp")

    # f1 vs f2: f1 is better (10 > 5). f2 rejected.

    assert mask[0], "f1 should be kept"
    assert not mask[1], "f2 should be rejected"
    assert mask[2], "f3 should be kept"
    assert not mask[3], "f4 should be rejected"

def test_dedup_disabled(sample_frames_for_dedup, mock_thumbnail_manager):
    filters = {"enable_dedup": False}
    config = Config()
    mask, reasons = _apply_deduplication_filter(sample_frames_for_dedup, filters, mock_thumbnail_manager, config, "/tmp")
    assert np.all(mask)
    assert not reasons

def test_dedup_threshold(sample_frames_for_dedup, mock_thumbnail_manager):
    # Set threshold to 0 (exact match only)
    filters = {"enable_dedup": True, "dedup_method": "pHash", "dedup_thresh": 0}
    config = Config()

    # f4 (1 bit diff) should NOT be rejected against f3
    mask, reasons = _apply_deduplication_filter(sample_frames_for_dedup, filters, mock_thumbnail_manager, config, "/tmp")

    assert not mask[0] # f1 and f2 are exact duplicates, f2 is better
    assert mask[1]
    assert mask[2]
    assert mask[3], "f4 should be kept because thresh is 0 and it has distance 1"
