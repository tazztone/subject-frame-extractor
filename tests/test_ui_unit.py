from unittest.mock import MagicMock, patch
from collections import deque

import pytest

from core.events import ExtractionEvent
from core.pipelines import execute_extraction
from ui.app_ui import AppUI


@pytest.fixture
def app_ui(
    mock_config, mock_logger, mock_progress_queue, mock_cancel_event, mock_thumbnail_manager, mock_model_registry
):
    with patch("ui.app_ui.AppUI.preload_models"):
        app = AppUI(
            mock_config,
            mock_logger,
            mock_progress_queue,
            mock_cancel_event,
            mock_thumbnail_manager,
            mock_model_registry,
        )
        app.components = {
            "face_seeding_group": MagicMock(),
            "text_seeding_group": MagicMock(),
            "auto_seeding_group": MagicMock(),
            "enable_face_filter_input": MagicMock(),
        }
        app.ana_input_components = []
        return app




def test_run_extraction_wrapper(app_ui):
    from ui.app_ui import ApplicationState
    state = ApplicationState()
    with patch.object(app_ui, "_run_pipeline", return_value=iter([])) as mock_run_pipeline:
        # Args matching ext_ui_map_keys:
        # ['source_path', 'upload_video', 'method', 'interval', 'nth_frame', 'max_resolution', 'thumb_megapixels', 'scene_detect']
        args = ["video.mp4", None, "interval", 1.0, 1, "1080", 0.5, True]

        gen = app_ui.run_extraction_wrapper(state, *args)
        list(gen)

        mock_run_pipeline.assert_called_once()
        call_args = mock_run_pipeline.call_args[0]
        assert call_args[0] == execute_extraction
        event = call_args[1]
        assert isinstance(event, ExtractionEvent)
        assert event.source_path == "video.mp4"
        assert event.method == "interval"


def test_fix_strategy_visibility_face_ref(app_ui):
    res = app_ui._fix_strategy_visibility("Face (Reference)")
    assert isinstance(res, dict)
    assert len(res) > 0


def test_fix_strategy_visibility_text(app_ui):
    res = app_ui._fix_strategy_visibility("Text Prompt")
    assert isinstance(res, dict)


def test_get_metric_description(app_ui):
    desc = app_ui.get_metric_description("quality_score")
    assert "score" in desc


# ============================================================================
# Phase 0 Tests: Min Confidence Filter, Text Strategy Warning
# ============================================================================


class TestMinConfidenceFilter:
    """Tests for the Min Confidence filter fix (Issue #1)."""

    def test_scene_without_score_is_filtered_when_threshold_positive(self, app_ui):
        """Scenes without score should be filtered when min_confidence > 0.

        This tests the fix where score defaults to 0 instead of 100.
        """

        # Create scene with no score in seed_metrics
        scenes = [
            {
                "shot_id": 1,
                "start_frame": 0,
                "end_frame": 50,
                "seed_metrics": {},  # No score!
                "seed_result": {"details": {"mask_area_pct": 50}},
                "status": "included",
                "manual_status_change": False,
                "rejection_reasons": [],
            }
        ]

        # Mock logger and other dependencies
        app_ui.logger = MagicMock()

        with patch("ui.app_ui.save_scene_seeds"):
            with patch("ui.app_ui.build_scene_gallery_items", return_value=([], [], 1)):
                with patch("ui.app_ui.get_scene_status_text", return_value=("Status", MagicMock())):
                    result = app_ui.scene_handler.on_apply_bulk_scene_filters_extended(
                        scenes=scenes,
                        min_mask_pct=0.0,
                        min_face_sim=0.0,
                        min_quality=0.5,  # Set threshold > 0
                        enable_face_filter=False,
                        output_dir="/tmp/test",
                        view="All",
                        history=deque(),
                    )

        # Scene should be excluded because score defaults to 0, which is < 0.5
        updated_scenes = result[0]
        assert updated_scenes[0]["status"] == "excluded", (
            "Scene without score should be excluded when min_quality_score > 0"
        )
        assert "Quality" in updated_scenes[0]["rejection_reasons"][0], "Rejection reason should include 'Quality'"

    def test_scene_with_high_score_is_kept(self, app_ui):
        """Scenes with score >= threshold should be kept."""

        scenes = [
            {
                "shot_id": 1,
                "start_frame": 0,
                "end_frame": 50,
                "seed_metrics": {"quality_score": 0.8},  # High score
                "seed_result": {"details": {"mask_area_pct": 50}},
                "status": "included",
                "manual_status_change": False,
                "rejection_reasons": [],
            }
        ]

        app_ui.logger = MagicMock()

        with patch("ui.app_ui.save_scene_seeds"):
            with patch("ui.app_ui.build_scene_gallery_items", return_value=([], [], 1)):
                with patch("ui.app_ui.get_scene_status_text", return_value=("Status", MagicMock())):
                    result = app_ui.scene_handler.on_apply_bulk_scene_filters_extended(
                        scenes=scenes,
                        min_mask_pct=0.0,
                        min_face_sim=0.0,
                        min_quality=0.5,
                        enable_face_filter=False,
                        output_dir="/tmp/test",
                        view="All",
                        history=deque(),
                    )

        updated_scenes = result[0]
        assert updated_scenes[0]["status"] == "included", "Scene with score >= threshold should be kept"

    def test_manual_override_not_affected_by_filters(self, app_ui):
        """Scenes with manual_status_change should not be auto-filtered."""
        scenes = [
            {
                "shot_id": 1,
                "start_frame": 0,
                "end_frame": 50,
                "seed_metrics": {},  # No score
                "seed_result": {},
                "status": "included",
                "manual_status_change": True,  # Manual override
                "rejection_reasons": [],
            }
        ]

        app_ui.logger = MagicMock()

        with patch("ui.app_ui.save_scene_seeds"):
            with patch("ui.app_ui.build_scene_gallery_items", return_value=([], [], 1)):
                with patch("ui.app_ui.get_scene_status_text", return_value=("Status", MagicMock())):
                    result = app_ui.scene_handler.on_apply_bulk_scene_filters_extended(
                        scenes=scenes,
                        min_mask_pct=0.0,
                        min_face_sim=0.0,
                        min_quality=0.9,  # High threshold
                        enable_face_filter=False,
                        output_dir="/tmp/test",
                        view="All",
                        history=deque(),
                    )

        updated_scenes = result[0]
        assert updated_scenes[0]["status"] == "included", "Manual override scenes should not be auto-filtered"


class TestTextStrategyWarning:
    """Tests for TEXT strategy warning label (Issue #3)."""

    def test_text_strategy_has_warning_in_choices(self, app_ui):
        """TEXT strategy choice should include warning indicator."""
        choices = app_ui.PRIMARY_SEED_STRATEGY_CHOICES
        text_choice = [c for c in choices if "Text" in c][0]

        assert "⚠️" in text_choice or "Limited" in text_choice, "TEXT strategy should have warning indicator"
