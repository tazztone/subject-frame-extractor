
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch, ANY
from collections import defaultdict, Counter
from pathlib import Path
from core.filtering import (
    load_and_prep_filter_data,
    histogram_svg,
    build_all_metric_svgs,
    _extract_metric_arrays,
    _run_batched_lpips,
    _apply_deduplication_filter,
    _apply_metric_filters,
    apply_all_filters_vectorized,
    apply_ssim_dedup,
    apply_lpips_dedup
)

class TestFiltering:

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.filter_default_yaw = {'min': -45, 'max': 45}
        config.filter_default_pitch = {'min': -45, 'max': 45}
        config.model_dump.return_value = {}
        config.filter_default_quality_score = {'default_min': 0.0, 'default_max': 100.0}
        config.filter_default_face_sim = {'default_min': 0.0, 'default_max': 1.0}
        config.filter_default_mask_area_pct = {'default_min': 0.0}
        config.filter_default_eyes_open = {'default_min': 0.0}
        return config

    @pytest.fixture
    def mock_thumbnail_manager(self):
        return MagicMock()

    @pytest.fixture
    def sample_frames(self):
        return [
            {'filename': 'frame_001.png', 'metrics': {'quality_score': 80, 'yaw': 0, 'pitch': 0}, 'phash': '0000000000000000'},
            {'filename': 'frame_002.png', 'metrics': {'quality_score': 90, 'yaw': 10, 'pitch': 10}, 'phash': 'FFFFFFFFFFFFFFFF'},
            {'filename': 'frame_003.png', 'metrics': {'quality_score': 80, 'yaw': 0, 'pitch': 0}, 'phash': '0000000000000000'}, # Duplicate of 001
        ]

    # --- Load & Prep Data ---

    @patch('core.filtering.Database')
    @patch('pathlib.Path.exists', return_value=True)
    def test_load_and_prep_filter_data(self, mock_exists, mock_db_cls, mock_config, sample_frames):
        db_mock = mock_db_cls.return_value
        db_mock.load_all_metadata.return_value = sample_frames

        get_keys = lambda: ['quality_score', 'yaw']

        frames, metrics = load_and_prep_filter_data("/tmp/out", get_keys, mock_config)

        assert len(frames) == 3
        assert 'quality_score' in metrics
        assert 'quality_score_hist' in metrics
        assert len(metrics['quality_score']) == 3

    # --- Visuals ---

    @patch('core.filtering.plt')
    def test_histogram_svg(self, mock_plt, mock_logger):
        hist_data = ([1, 2], [0, 50, 100])

        # Mock plt context manager
        mock_fig = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, MagicMock())
        mock_plt.style.context.return_value.__enter__.return_value = None

        svg = histogram_svg(hist_data, "Title", mock_logger)

        assert svg == ""
        mock_fig.savefig.assert_called()

    # --- Array Extraction ---

    def test_extract_metric_arrays(self, sample_frames, mock_config):
        arrays = _extract_metric_arrays(sample_frames, mock_config)

        assert 'quality_score' in arrays
        assert len(arrays['quality_score']) == 3
        assert arrays['quality_score'][0] == 80
        assert arrays['quality_score'][1] == 90

    # --- LPIPS Batching ---

    @patch('core.filtering.get_lpips_metric')
    @patch('torch.stack')
    def test_run_batched_lpips(self, mock_stack, mock_get_lpips, sample_frames, mock_thumbnail_manager):
        # Setup
        dedup_mask = np.array([True, True, True])
        reasons = defaultdict(list)
        pairs = [(0, 1)] # Compare frame 1 (80) and 2 (90)

        # Mock LPIPS model
        mock_model = MagicMock()
        mock_get_lpips.return_value = mock_model
        # Return distance 0.05 (duplicate)
        mock_model.forward.return_value.squeeze.return_value.cpu.return_value.numpy.return_value = np.array([0.05])

        # Mock images
        # Ensure thumbnail_manager.get returns something NOT None, otherwise loop continues
        mock_thumbnail_manager.get.return_value = np.zeros((10,10,3), dtype=np.uint8)

        _run_batched_lpips(pairs, sample_frames, dedup_mask, reasons, mock_thumbnail_manager, "/tmp/out", 0.1)

        # If thumbnail_manager.get was called correctly, dedup logic runs.
        # Check calls
        assert mock_thumbnail_manager.get.called

        # Frame 1 (80) vs Frame 2 (90).
        # c_score > p_score. p_idx=0 is duplicate.
        # But wait, did it modify dedup_mask?

        # If dedup_mask is passed as reference, it should be modified.
        # If assertion fails, maybe valid_indices was empty?
        # Check Path logic: `Path(output_dir) / "thumbs" / filename`.
        # Path("/tmp/out") / "thumbs" / "frame_001.png"

        assert dedup_mask[0] is False
        assert dedup_mask[1] is True
        assert 'duplicate' in reasons['frame_001.png']

    # --- Deduplication ---

    def test_apply_deduplication_filter_phash(self, sample_frames, mock_config):
        filters = {"enable_dedup": True, "dedup_method": "pHash", "dedup_thresh": 0}

        mask, reasons = _apply_deduplication_filter(sample_frames, filters, None, mock_config, "/tmp/out")

        assert mask[0] is True
        assert mask[1] is True
        assert mask[2] is False
        assert 'duplicate' in reasons['frame_003.png']

    @patch('core.filtering._run_batched_lpips')
    def test_apply_deduplication_filter_lpips(self, mock_run_lpips, sample_frames, mock_config, mock_thumbnail_manager):
        filters = {"enable_dedup": True, "dedup_method": "LPIPS", "lpips_threshold": 0.1}

        mask, reasons = _apply_deduplication_filter(sample_frames, filters, mock_thumbnail_manager, mock_config, "/tmp/out")

        mock_run_lpips.assert_called()

    # --- Metric Filtering ---

    def test_apply_metric_filters(self, sample_frames, mock_config):
        arrays = _extract_metric_arrays(sample_frames, mock_config)
        filters = {"quality_score_min": 85}

        mask, reasons = _apply_metric_filters(sample_frames, arrays, filters, mock_config)

        assert mask[0] is False
        assert mask[1] is True
        assert mask[2] is False
        assert 'quality_score_low' in reasons['frame_001.png']

    # --- Full Pipeline ---

    def test_apply_all_filters_vectorized(self, sample_frames, mock_config):
        filters = {
            "enable_dedup": True,
            "dedup_method": "pHash",
            "dedup_thresh": 0,
            "quality_score_min": 85
        }

        kept, rejected, stats, reasons = apply_all_filters_vectorized(sample_frames, filters, mock_config, None, "/tmp/out")

        assert len(kept) == 1
        assert kept[0]['filename'] == 'frame_002.png'
        assert len(rejected) == 2

        assert 'quality_score_low' in reasons['frame_001.png']
        assert 'duplicate' in reasons['frame_003.png']
