
import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path
from ui.gallery_utils import (
    _update_gallery,
    on_filters_changed,
    auto_set_thresholds
)
from core.events import FilterEvent

class TestGalleryUtils:

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        # Setup config defaults if needed by apply_all_filters_vectorized
        config.model_dump.return_value = {}
        config.filter_default_quality_score = {'default_min': 0.0, 'default_max': 100.0}
        config.filter_default_face_sim = {'default_min': 0.0, 'default_max': 1.0}
        return config

    @pytest.fixture
    def mock_thumbnail_manager(self):
        tm = MagicMock()
        tm.get.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        return tm

    @pytest.fixture
    def sample_frames_data(self):
        return [
            {'filename': 'f1.webp', 'metrics': {'quality_score': 90}},
            {'filename': 'f2.webp', 'metrics': {'quality_score': 80}},
            {'filename': 'f3.webp', 'metrics': {'quality_score': 70}},
        ]

    # --- _update_gallery ---

    @patch('ui.gallery_utils.apply_all_filters_vectorized')
    @patch('ui.gallery_utils.render_mask_overlay')
    def test_update_gallery_kept(self, mock_render, mock_apply, sample_frames_data, mock_thumbnail_manager, mock_config, mock_logger):
        # Mock apply to return frames 1 and 2 kept
        kept = sample_frames_data[:2]
        rejected = [sample_frames_data[2]]
        counts = MagicMock()
        counts.most_common.return_value = [('low_quality', 1)]
        reasons = {'f3.webp': ['low_quality']}

        mock_apply.return_value = (kept, rejected, counts, reasons)

        filters = {}
        output_dir = "/tmp/out"
        gallery_view = "Kept Frames"

        status, update = _update_gallery(
            sample_frames_data, filters, output_dir, gallery_view,
            False, 0.5, mock_thumbnail_manager, mock_config, mock_logger
        )

        assert "Kept:** 2/3" in status
        assert "Rejections:** low_quality: 1" in status
        assert len(update['value']) == 2
        mock_thumbnail_manager.get.assert_called()

    @patch('ui.gallery_utils.apply_all_filters_vectorized')
    def test_update_gallery_rejected(self, mock_apply, sample_frames_data, mock_thumbnail_manager, mock_config, mock_logger):
        kept = sample_frames_data[:2]
        rejected = [sample_frames_data[2]]
        reasons = {'f3.webp': ['low_quality']}
        mock_apply.return_value = (kept, rejected, MagicMock(), reasons)

        status, update = _update_gallery(
            sample_frames_data, {}, "/tmp/out", "Rejected",
            False, 0.5, mock_thumbnail_manager, mock_config, mock_logger
        )

        assert len(update['value']) == 1
        # Check caption contains reason
        assert "low_quality" in update['value'][0][1]

    @patch('ui.gallery_utils.apply_all_filters_vectorized')
    @patch('cv2.imread')
    @patch('ui.gallery_utils.render_mask_overlay')
    @patch('pathlib.Path.exists', return_value=True)
    def test_update_gallery_overlay(self, mock_exists, mock_render, mock_imread, mock_apply, sample_frames_data, mock_thumbnail_manager, mock_config, mock_logger):
        # Add mask data to first frame
        sample_frames_data[0]['mask_path'] = 'mask.png'
        sample_frames_data[0]['mask_empty'] = False

        kept = [sample_frames_data[0]]
        mock_apply.return_value = (kept, [], MagicMock(), {})

        mock_render.return_value = np.ones((100, 100, 3), dtype=np.uint8)

        status, update = _update_gallery(
            sample_frames_data, {}, "/tmp/out", "Kept",
            True, 0.5, mock_thumbnail_manager, mock_config, mock_logger
        )

        mock_render.assert_called()
        mock_imread.assert_called()

    # --- on_filters_changed ---

    @patch('ui.gallery_utils._update_gallery')
    def test_on_filters_changed(self, mock_update, sample_frames_data, mock_thumbnail_manager, mock_config, mock_logger):
        mock_update.return_value = ("Status", "Update")

        event = FilterEvent(
            all_frames_data=sample_frames_data,
            slider_values={'quality_score_min': 80},
            require_face_match=False,
            dedup_thresh=5,
            dedup_method="pHash",
            per_metric_values={'face_sim': [0.5]}, # Enables face_sim filter flag
            output_dir="/tmp/out",
            gallery_view="Kept",
            show_overlay=False,
            overlay_alpha=0.5
        )

        res = on_filters_changed(event, mock_thumbnail_manager, mock_config, mock_logger)

        assert res['filter_status_text'] == "Status"
        assert res['results_gallery'] == "Update"

        # Verify filters dict passed to _update_gallery
        call_args = mock_update.call_args
        filters_arg = call_args[0][1]
        assert filters_arg['quality_score_min'] == 80
        assert filters_arg['face_sim_enabled'] is True
        # Check dedup enablement logic: any('phash'...)
        # sample_frames_data has no phash, so enable_dedup should be False
        assert filters_arg['enable_dedup'] is False

    def test_on_filters_changed_empty(self, mock_thumbnail_manager, mock_config, mock_logger):
        event = FilterEvent(
            all_frames_data=[],
            slider_values={},
            require_face_match=False,
            dedup_thresh=0,
            dedup_method="",
            per_metric_values={},
            output_dir="",
            gallery_view="",
            show_overlay=False,
            overlay_alpha=0.0
        )
        res = on_filters_changed(event, mock_thumbnail_manager, mock_config, mock_logger)
        assert "Run analysis" in res['filter_status_text']

    # --- auto_set_thresholds ---

    def test_auto_set_thresholds(self):
        per_metric_values = {
            'quality_score': [10, 20, 30, 40, 50], # p50 = 30
            'sharpness': [100, 200, 300], # p50 = 200
            'ignore_me': []
        }
        slider_keys = ['quality_score_min', 'sharpness_min', 'other_min']
        selected_metrics = ['quality_score', 'sharpness']

        updates = auto_set_thresholds(per_metric_values, 50, slider_keys, selected_metrics)

        assert updates['slider_quality_score_min']['value'] == 30.0
        assert updates['slider_sharpness_min']['value'] == 200.0
        # other_min not in metrics
        assert 'value' not in updates.get('slider_other_min', {}) or updates['slider_other_min'].get('value') is None

    def test_auto_set_thresholds_empty(self):
        updates = auto_set_thresholds({}, 50, ['k'], [])
        assert 'value' not in updates['slider_k']
