import pytest
import sys
from unittest.mock import patch, MagicMock, ANY
import numpy as np
import gradio as gr

# Mock heavy dependencies before they are imported by the app
mock_imagehash = MagicMock()
modules_to_mock = {
    'imagehash': mock_imagehash,
    'matplotlib': MagicMock(),
    'matplotlib.pyplot': MagicMock(),
    'app.frames': MagicMock(),
}
patch.dict(sys.modules, modules_to_mock).start()

from app.filter_logic import (
    apply_all_filters_vectorized,
    auto_set_thresholds,
    reset_filters,
)
from app.config import Config

# --- Test Data and Mocks ---

@pytest.fixture
def mock_config():
    """Fixture to provide a mock Config object for tests."""
    with patch('app.filter_logic.Config') as MockConfig:
        mock_instance = MockConfig.return_value
        mock_instance.QUALITY_METRICS = ['sharpness', 'contrast']
        mock_instance.filter_defaults = {
            'sharpness': {'default_min': 10, 'default_max': 90},
            'contrast': {'default_min': 20, 'default_max': 80},
            'dedup_thresh': {'default': 5},
        }
        mock_instance.ui_defaults = {'require_face_match': False}
        yield mock_instance

@pytest.fixture
def sample_frames_data():
    """Provides a list of sample frame metadata for testing."""
    return [
        # Kept frame
        {'filename': 'frame_01.png', 'phash': 'a'*16, 'metrics': {'sharpness_score': 50, 'contrast_score': 50}, 'face_sim': 0.8, 'mask_area_pct': 20},
        # Rejected by sharpness
        {'filename': 'frame_02.png', 'phash': 'b'*16, 'metrics': {'sharpness_score': 5, 'contrast_score': 50}, 'face_sim': 0.8, 'mask_area_pct': 20},
        # Rejected by face similarity
        {'filename': 'frame_03.png', 'phash': 'c'*16, 'metrics': {'sharpness_score': 50, 'contrast_score': 50}, 'face_sim': 0.2, 'mask_area_pct': 20},
        # Rejected by mask area
        {'filename': 'frame_04.png', 'phash': 'd'*16, 'metrics': {'sharpness_score': 50, 'contrast_score': 50}, 'face_sim': 0.8, 'mask_area_pct': 2},
        # Duplicate of frame_01
        {'filename': 'frame_05.png', 'phash': 'a'*16, 'metrics': {'sharpness_score': 50, 'contrast_score': 50}, 'face_sim': 0.8, 'mask_area_pct': 20},
        # Frame with missing face sim data
        {'filename': 'frame_06.png', 'phash': 'e'*16, 'metrics': {'sharpness_score': 50, 'contrast_score': 50}, 'mask_area_pct': 20},
    ]

# --- Tests for apply_all_filters_vectorized ---

def test_apply_all_filters_no_filters(sample_frames_data, mock_config):
    """Test that with no filters, all frames are kept."""
    kept, rejected, _, _ = apply_all_filters_vectorized(sample_frames_data, {})
    assert len(kept) == len(sample_frames_data)
    assert len(rejected) == 0

def test_apply_quality_filters(sample_frames_data, mock_config):
    """Test filtering based on quality metrics like sharpness."""
    filters = {'sharpness_min': 10, 'contrast_min': 10}
    kept, rejected, _, reasons = apply_all_filters_vectorized(sample_frames_data, filters)
    assert len(kept) == 5
    assert len(rejected) == 1
    assert rejected[0]['filename'] == 'frame_02.png'
    assert 'sharpness_low' in reasons['frame_02.png']

def test_apply_face_similarity_filter(sample_frames_data, mock_config):
    """Test filtering based on face similarity score."""
    filters = {'face_sim_enabled': True, 'face_sim_min': 0.5}
    kept, rejected, _, reasons = apply_all_filters_vectorized(sample_frames_data, filters)
    assert len(kept) == 5
    assert len(rejected) == 1
    assert rejected[0]['filename'] == 'frame_03.png'
    assert 'face_sim_low' in reasons['frame_03.png']

def test_apply_face_filter_require_match(sample_frames_data, mock_config):
    """Test rejecting frames that are missing a face when required."""
    filters = {'face_sim_enabled': True, 'require_face_match': True, 'face_sim_min': 0.5}
    kept, rejected, _, reasons = apply_all_filters_vectorized(sample_frames_data, filters)
    # Rejects frame_03 (low sim) and frame_06 (missing sim)
    assert len(kept) == 4
    assert len(rejected) == 2
    assert 'frame_06.png' in [r['filename'] for r in rejected]
    assert 'face_missing' in reasons['frame_06.png']

def test_apply_mask_area_filter(sample_frames_data, mock_config):
    """Test filtering based on mask area percentage."""
    filters = {'mask_area_enabled': True, 'mask_area_pct_min': 10}
    kept, rejected, _, reasons = apply_all_filters_vectorized(sample_frames_data, filters)
    assert len(kept) == 5
    assert len(rejected) == 1
    assert rejected[0]['filename'] == 'frame_04.png'
    assert 'mask_too_small' in reasons['frame_04.png']

def test_apply_deduplication_filter(sample_frames_data, mock_config):
    """Test filtering of duplicate frames using phash."""
    # This mock now correctly simulates the behavior of imagehash
    mock_hash_a = MagicMock()
    mock_hash_a.hash = 'a'*16
    mock_hash_a.__sub__ = lambda s, other: 0 if s.hash == other.hash else 10

    mock_hash_b = MagicMock()
    mock_hash_b.hash = 'b'*16
    mock_hash_b.__sub__ = lambda s, other: 0 if s.hash == other.hash else 10

    mock_hashes = {
        'a'*16: mock_hash_a,
        'b'*16: mock_hash_b,
        'c'*16: MagicMock(__sub__=lambda s, other: 10),
        'd'*16: MagicMock(__sub__=lambda s, other: 10),
        'e'*16: MagicMock(__sub__=lambda s, other: 10),
    }

    mock_imagehash.hex_to_hash.side_effect = lambda h: mock_hashes[h]

    filters = {'enable_dedup': True, 'dedup_thresh': 5}
    kept, rejected, _, reasons = apply_all_filters_vectorized(sample_frames_data, filters)

    assert len(kept) == 5
    assert len(rejected) == 1
    assert rejected[0]['filename'] == 'frame_05.png'
    assert 'duplicate' in reasons['frame_05.png']

# --- Tests for auto_set_thresholds ---

def test_auto_set_thresholds():
    """Test that thresholds are correctly calculated based on percentiles."""
    per_metric_values = {
        'sharpness': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], # 75th pctl = 77.5
        'contrast': [1, 2, 3, 4, 5], # 75th pctl = 4.0
    }
    slider_keys = ['sharpness_min', 'sharpness_max', 'contrast_min']

    updates = auto_set_thresholds(per_metric_values, 75, slider_keys)

    assert updates['slider_sharpness_min']['value'] == 77.5
    assert updates['slider_contrast_min']['value'] == 4.0
    # max sliders should not be updated
    assert 'value' not in updates['slider_sharpness_max']

def test_auto_set_thresholds_no_data():
    """Test that it returns empty updates when there is no data."""
    slider_keys = ['sharpness_min']
    updates = auto_set_thresholds({}, 90, slider_keys)
    assert 'value' not in updates['slider_sharpness_min']

# --- Tests for reset_filters ---

@patch('app.filter_logic.on_filters_changed')
def test_reset_filters(mock_on_filters_changed, mock_config):
    """Test that all filter sliders are reset to their default values."""
    mock_on_filters_changed.return_value = {
        'filter_status_text': 'status',
        'results_gallery': []
    }
    slider_keys = ['sharpness_min', 'sharpness_max', 'contrast_min']

    updates = reset_filters([], {}, '/fake/dir', mock_config, slider_keys, MagicMock())

    assert updates['slider_sharpness_min']['value'] == 10
    assert updates['slider_sharpness_max']['value'] == 90
    assert updates['slider_contrast_min']['value'] == 20
    assert updates['dedup_thresh_input']['value'] == 5
    assert updates['require_face_match_input']['value'] == False

    # Check that on_filters_changed is NOT called when there's no data
    mock_on_filters_changed.assert_not_called()
    assert updates['filter_status_text'] == "Load an analysis to begin."