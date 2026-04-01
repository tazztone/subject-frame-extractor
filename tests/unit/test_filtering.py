from collections import defaultdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.filtering import (
    _apply_metric_filters,
    _extract_metric_arrays,
    apply_all_filters_vectorized,
    load_and_prep_filter_data,
)
from core.operators.dedup import _run_batched_lpips, apply_deduplication_filter
from core.operators.viz import histogram_svg


class TestFiltering:
    @pytest.fixture
    def mock_thumbnail_manager(self):
        return MagicMock()

    @pytest.fixture
    def sample_frames(self):
        return [
            {
                "filename": "frame_001.png",
                "metrics": {"quality_score": 80, "yaw": 0, "pitch": 0},
                "phash": "0000000000000000",
            },
            {
                "filename": "frame_002.png",
                "metrics": {"quality_score": 90, "yaw": 10, "pitch": 10},
                "phash": "FFFFFFFFFFFFFFFF",
            },
            {
                "filename": "frame_003.png",
                "metrics": {"quality_score": 80, "yaw": 0, "pitch": 0},
                "phash": "0000000000000000",
            },  # Duplicate of 001
        ]

    # --- Load & Prep Data ---

    @patch("core.filtering.Database")
    @patch("pathlib.Path.exists", return_value=True)
    def test_load_and_prep_filter_data(self, mock_exists, mock_db_cls, mock_config, sample_frames):
        db_mock = mock_db_cls.return_value
        db_mock.load_all_metadata.return_value = sample_frames

        def get_keys():
            return ["quality_score", "yaw"]

        frames, metrics = load_and_prep_filter_data("/tmp/out", get_keys, mock_config)

        assert len(frames) == 3
        assert "quality_score" in metrics
        assert "quality_score_hist" in metrics
        assert len(metrics["quality_score"]) == 3

    # --- Visuals ---

    @patch("core.operators.viz.plt")
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

        assert "quality_score" in arrays
        assert len(arrays["quality_score"]) == 3
        assert arrays["quality_score"][0] == 80
        assert arrays["quality_score"][1] == 90

    # --- LPIPS Batching ---

    @patch("core.operators.dedup.get_lpips_metric")
    @patch("torch.stack")
    def test_run_batched_lpips(self, mock_stack, mock_get_lpips, sample_frames, mock_thumbnail_manager):
        # Setup
        dedup_mask = np.array([True, True, True])
        reasons = defaultdict(list)
        pairs = [(0, 1)]  # Compare frame 1 (80) and 2 (90)

        # Mock LPIPS model
        mock_model = MagicMock()
        mock_get_lpips.return_value = mock_model
        # Return distance 0.05 (duplicate)
        mock_model.forward.return_value.squeeze.return_value.cpu.return_value.numpy.return_value = np.array([0.05])

        # Mock images
        mock_thumbnail_manager.get.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

        _run_batched_lpips(pairs, sample_frames, dedup_mask, reasons, mock_thumbnail_manager, "/tmp/out", 0.1)

        assert mock_thumbnail_manager.get.called
        assert dedup_mask[0] == False
        assert dedup_mask[1] == True
        assert "duplicate" in reasons["frame_001.png"]

    # --- Deduplication ---

    def test_apply_deduplication_filter_phash(self, sample_frames, mock_config):
        filters = {"enable_dedup": True, "dedup_method": "pHash", "dedup_thresh": 0}

        mask, reasons = apply_deduplication_filter(sample_frames, filters, None, mock_config, "/tmp/out")

        assert mask[0] == True
        assert mask[1] == True
        assert mask[2] == False
        assert "duplicate" in reasons["frame_003.png"]

    @patch("core.operators.dedup._run_batched_lpips")
    def test_apply_deduplication_filter_lpips(self, mock_run_lpips, sample_frames, mock_config, mock_thumbnail_manager):
        filters = {"enable_dedup": True, "dedup_method": "LPIPS", "lpips_threshold": 0.1}

        mask, reasons = apply_deduplication_filter(
            sample_frames, filters, mock_thumbnail_manager, mock_config, "/tmp/out"
        )

        mock_run_lpips.assert_called()

    # --- Metric Filtering ---

    def test_apply_metric_filters(self, sample_frames, mock_config):
        arrays = _extract_metric_arrays(sample_frames, mock_config)
        filters = {"quality_score_min": 85}

        mask, reasons = _apply_metric_filters(sample_frames, arrays, filters, mock_config)

        assert mask[0] == False
        assert mask[1] == True
        assert mask[2] == False
        assert "quality_score_low" in reasons["frame_001.png"]

    # --- Full Pipeline ---

    def test_apply_all_filters_vectorized_combined(self, sample_frames, mock_config):
        """Test combined quality + mask-area + dedup filter."""
        # Add mask_area_pct to frames
        for f in sample_frames:
            f["mask_area_pct"] = 10.0
        sample_frames[0]["mask_area_pct"] = 1.0  # Too low

        filters = {
            "enable_dedup": True,
            "dedup_method": "pHash",
            "dedup_thresh": 0,
            "quality_score_min": 70,
            "mask_area_enabled": True,
            "mask_area_pct_min": 5.0,
        }

        kept, rejected, stats, reasons = apply_all_filters_vectorized(
            sample_frames, filters, mock_config, None, "/tmp/out"
        )

        assert len(kept) == 1
        assert kept[0]["filename"] == "frame_002.png"
        assert "mask_too_small" in reasons["frame_001.png"]
        assert "duplicate" in reasons["frame_003.png"]

    def test_apply_all_filters_all_filtered_out(self, sample_frames, mock_config):
        """Test edge case where all frames are filtered out."""
        filters = {"quality_score_min": 100}
        kept, rejected, stats, reasons = apply_all_filters_vectorized(
            sample_frames, filters, mock_config, None, "/tmp/out"
        )
        assert len(kept) == 0
        assert len(rejected) == 3

    def test_apply_all_filters_face_sim_edge_cases(self, sample_frames, mock_config):
        """Test face_sim filtering with require_face_match."""
        sample_frames[0]["face_sim"] = 0.8
        sample_frames[1]["face_sim"] = 0.4
        sample_frames[2]["face_sim"] = np.nan

        filters = {"face_sim_enabled": True, "face_sim_min": 0.5, "require_face_match": True}

        kept, rejected, stats, reasons = apply_all_filters_vectorized(
            sample_frames, filters, mock_config, None, "/tmp/out"
        )

        # f1: 0.8 >= 0.5 -> kept
        # f2: 0.4 < 0.5 -> rejected
        # f3: nan and require_face_match -> rejected
        assert len(kept) == 1
        assert kept[0]["filename"] == "frame_001.png"
        assert "face_sim_low" in reasons["frame_002.png"]
        assert "face_missing" in reasons["frame_003.png"]

    def test_apply_all_filters_range_filter(self, sample_frames, mock_config):
        """Test a range-based filter (e.g., yaw)."""
        sample_frames[0]["metrics"]["yaw"] = -10
        sample_frames[1]["metrics"]["yaw"] = 0
        sample_frames[2]["metrics"]["yaw"] = 20

        # Mock config to define range filter for yaw
        mock_config.filter_default_yaw = {"type": "range", "default_min": -5, "default_max": 5}
        # Ensure 'yaw' is in the quality weight keys or recognized
        mock_config.model_dump.return_value = {"quality_weights_yaw": 1}

        filters = {"yaw_min": -5, "yaw_max": 5}

        kept, rejected, stats, reasons = apply_all_filters_vectorized(
            sample_frames, filters, mock_config, None, "/tmp/out"
        )

        # f1: -10 < -5 -> rejected
        # f2: 0 -> kept
        # f3: 20 > 5 -> rejected
        assert len(kept) == 1
        assert kept[0]["filename"] == "frame_002.png"
