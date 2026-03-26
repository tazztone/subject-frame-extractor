from queue import Queue
from unittest.mock import MagicMock, patch

import gradio as gr
import pytest

from ui.app_ui import ApplicationState, AppUI


class TestAppUI:
    @pytest.fixture
    def mock_queue(self):
        return Queue()

    @pytest.fixture
    def app_state(self):
        return ApplicationState()

    @pytest.fixture
    def app_ui(
        self, mock_config, mock_logger, mock_queue, mock_cancel_event, mock_thumbnail_manager, mock_model_registry
    ):
        # We need to mock torch.cuda.is_available inside AppUI init
        with patch("torch.cuda.is_available", return_value=False):
            ui = AppUI(
                mock_config, mock_logger, mock_queue, mock_cancel_event, mock_thumbnail_manager, mock_model_registry
            )

        # Manually populate components usually created by build_ui
        ui.components = {
            "metric_sliders": {},
            "metric_accs": {},
            "metric_plots": {},
            "metric_auto_threshold_cbs": {},
            "cancel_button": MagicMock(),
            "pause_button": MagicMock(),
            "unified_status": MagicMock(),
            "unified_log": MagicMock(),
            "main_tabs": MagicMock(),
            "stepper": MagicMock(),
            "seeding_results_column": MagicMock(),
            "propagation_group": MagicMock(),
            "propagate_masks_button": MagicMock(),
            "scene_filter_status": MagicMock(),
            "scene_face_sim_min_input": MagicMock(),
            "scene_gallery": MagicMock(),
            "scene_gallery_index_map_state": MagicMock(),
            "analysis_metadata_path_state": MagicMock(),
            "filtering_tab": MagicMock(),
            "source_input": MagicMock(),
            "max_resolution": MagicMock(),
            "thumb_megapixels_input": MagicMock(),
            "ext_scene_detect_input": MagicMock(),
            "method_input": MagicMock(),
            "pre_analysis_enabled_input": MagicMock(),
            "pre_sample_nth_input": MagicMock(),
            "enable_face_filter_input": MagicMock(),
            "face_model_name_input": MagicMock(),
            "face_ref_img_path_input": MagicMock(),
            "text_prompt_input": MagicMock(),
            "best_frame_strategy_input": MagicMock(),
            "tracker_model_name_input": MagicMock(),
            "primary_seed_strategy_input": MagicMock(),
            "filter_status_text": MagicMock(),
            "results_gallery": MagicMock(),
            "all_frames_data_state": MagicMock(),
            "per_metric_values_state": MagicMock(),
            "results_group": MagicMock(),
            "export_group": MagicMock(),
            "dedup_method_input": MagicMock(),
            "show_mask_overlay_input": MagicMock(),
            "overlay_alpha_slider": MagicMock(),
            "require_face_match_input": MagicMock(),
            "dedup_thresh_input": MagicMock(),
            "smart_filter_state": MagicMock(),
            "smart_filter_checkbox": MagicMock(),
            "scene_history_state": MagicMock(),
            "application_state": MagicMock(),
            "face_seeding_group": MagicMock(),
            "text_seeding_group": MagicMock(),
            "auto_seeding_group": MagicMock(),
            "total_pages_label": MagicMock(),
            "page_number_input": MagicMock(),
            "sceneeditorstatusmd": MagicMock(),
            "sceneeditorpromptinput": MagicMock(),
            "scene_editor_group": MagicMock(),
            "subject_selection_gallery": MagicMock(),
            "gallery_image_preview": MagicMock(),
            "scene_mask_area_min_input": MagicMock(),
            "scene_quality_score_min_input": MagicMock(),
            "scene_gallery_columns": MagicMock(),
            "scene_gallery_height": MagicMock(),
        }

        # Populate metric sliders with string labels
        mock_slider = MagicMock()
        mock_slider.label = "Quality Score"
        ui.components["metric_sliders"]["quality_score_min"] = mock_slider

        return ui

    # --- Initialization ---

    def test_init(self, app_ui):
        assert app_ui.cuda_available is False
        assert app_ui.batch_manager is not None

    def test_preload_models(self, app_ui):
        with patch("threading.Thread") as mock_thread:
            app_ui.preload_models()
            mock_thread.assert_called_once()

    # --- UI Helpers ---

    def test_fix_strategy_visibility(self, app_ui):
        updates = app_ui._fix_strategy_visibility("Source Face Reference")
        assert updates[app_ui.components["face_seeding_group"]]["visible"] is True
        assert updates[app_ui.components["text_seeding_group"]]["visible"] is False

    # --- Pipeline Wrappers ---

    def test_run_extraction_wrapper(self, app_ui, app_state):
        # We assume _run_pipeline is working (tested elsewhere or mocked)
        # We test that it creates event correctly

        with patch.object(app_ui, "_run_pipeline") as mock_run:
            mock_run.return_value = iter([])
            # ext keys: source_path, upload_video, method, interval, nth_frame, max_resolution, thumb_megapixels, scene_detect
            args = ["vid.mp4", None, "interval", 1.0, 1, "1080", 0.5, True]

            # Consume generator
            list(app_ui.pipeline_handler.run_extraction_wrapper(app_state, *args))

            mock_run.assert_called_once()
            # Verify event creation
            event = mock_run.call_args[0][1]
            assert event.source_path == "vid.mp4"
            assert event.method == "interval"

    def test_run_pre_analysis_wrapper(self, app_ui, app_state, tmp_path):
        out_dir = str(tmp_path / "out")
        with patch.object(app_ui, "_run_pipeline") as mock_run:
            mock_run.return_value = iter([])
            # Ana keys: resume (index 0), then checks/params...
            # Removed: output_folder, video_path
            args = [
                False,  # resume
                False,  # enable_face_filter
                "",  # face_ref_img_path
                None,  # face_ref_img_upload
                "buffalo_l",  # face_model_name
                True,  # enable_subject_mask
                "sam3",  # tracker_model_name
                "Largest",  # best_frame_strategy
                True,  # scene_detect
                "",  # text_prompt
                1.0,  # min_mask_area_pct
                2500.0,  # sharpness_base_scale
                100.0,  # edge_strength_base_scale
                True,  # pre_analysis_enabled
                1,  # pre_sample_nth
                "Automatic Detection",  # primary_seed_strategy
                True,  # compute_quality_score
                True,  # compute_sharpness
            ] + [True] * 11  # remaining compute_... metrics

            app_state.extracted_video_path = "vid.mp4"
            app_state.analysis_output_dir = out_dir

            list(app_ui.pipeline_handler.run_pre_analysis_wrapper(app_state, *args))
            mock_run.assert_called_once()
            event = mock_run.call_args[0][1]
            assert event.output_folder == out_dir
            assert event.video_path == "vid.mp4"

    # --- Success Callbacks ---

    def test_on_extraction_success(self, app_ui, app_state):
        res = {"extracted_video_path_state": "v.mp4", "extracted_frames_dir_state": "/frames"}
        updates = app_ui.pipeline_handler._on_extraction_success(res, app_state)

        new_state = updates[app_ui.components["application_state"]]
        assert new_state.extracted_video_path == "v.mp4"
        assert "Extraction Complete" in updates[app_ui.components["unified_status"]]

    def test_on_pre_analysis_success(self, app_ui, app_state, tmp_path):
        scenes = [
            {
                "shot_id": 1,
                "start_frame": 0,
                "end_frame": 10,
                "status": "included",
                "seed_result": {"bbox": [0, 0, 10, 10]},
            }
        ]
        res = {"scenes": scenes, "output_dir": str(tmp_path / "out")}

        with patch("ui.handlers.pipeline_handlers.get_scene_status_text", return_value=("Status", "Button")):
            updates = app_ui.pipeline_handler._on_pre_analysis_success(res, app_state)

        new_state = updates[app_ui.components["application_state"]]
        assert new_state.scenes == scenes
        assert updates[app_ui.components["scene_filter_status"]] == "Status"

    # --- History/Undo ---

    def test_push_history(self, app_state):
        scenes = [{"id": 1}]
        app_state.push_history(scenes)
        assert len(app_state.scene_history) == 1
        assert app_state.scene_history[0] == scenes

    def test_undo_last_action(self, app_ui, app_state, tmp_path):
        out_dir = str(tmp_path / "out")
        app_state.extracted_frames_dir = out_dir
        app_state.push_history([{"shot_id": 1, "start_frame": 0, "end_frame": 10}])
        app_state.scenes = []

        with (
            patch("ui.handlers.scene_handler.save_scene_seeds"),
            patch("ui.handlers.scene_handler.build_scene_gallery_items", return_value=([], [], 1)),
            patch("ui.handlers.scene_handler.get_scene_status_text", return_value=("Stat", "Btn")),
        ):
            new_state, gal, msg = app_ui.scene_handler._undo_last_action(app_state, "Kept")

            assert len(new_state.scenes) == 1
            assert new_state.scenes[0]["shot_id"] == 1
            assert len(new_state.scene_history) == 0

    # --- Smart Mode Updates ---

    def test_get_smart_mode_updates(self, app_ui):
        updates = app_ui._get_smart_mode_updates(True)
        # Should return updates for sliders. quality_score_min is in mock components.
        # Check properties of update
        assert len(updates) == 1
        assert updates[0]["label"].endswith("(%)")

        updates_off = app_ui._get_smart_mode_updates(False)
        # Check that label DOES NOT end with (%)
        assert not updates_off[0]["label"].endswith("(%)")

    # --- Bulk Scene Filters ---

    def test_on_apply_bulk_scene_filters_extended(self, app_ui, app_state, tmp_path):
        out_dir = str(tmp_path / "out")
        app_state.extracted_frames_dir = out_dir
        app_state.scenes = [
            {
                "shot_id": 1,
                "start_frame": 0,
                "end_frame": 10,
                "status": "included",
                "seed_metrics": {"quality_score": 0.9},
                "seed_result": {"details": {"mask_area_pct": 50}},
                "manual_status_change": False,
            }
        ]

        with (
            patch("ui.handlers.scene_handler.save_scene_seeds"),
            patch("ui.handlers.scene_handler.build_scene_gallery_items", return_value=([], [], 1)),
        ):
            # Signature: on_apply_bulk_scene_filters_extended(self, app_state, min_mask_pct, min_face_sim, min_quality, enable_face_filter, view)
            new_state, _, _, _ = app_ui.scene_handler.on_apply_bulk_scene_filters_extended(
                app_state, 60.0, 0.0, 0.0, False, "Kept"
            )

            assert new_state.scenes[0]["status"] == "excluded"
            assert any("Area" in r for r in new_state.scenes[0]["rejection_reasons"])

    # --- Reset Filters ---

    def test_on_reset_filters(self, app_ui, app_state, tmp_path):
        res = app_ui.on_reset_filters(app_state)
        # tuple([new_state] + slider_updates + [5, False, "Filters Reset.", gr.update(), "Fast (pHash)"] + acc_updates + [False])

        assert res[0].smart_filter_enabled is False  # Smart filter state
        assert res[-1] is False  # Smart filter checkbox
        assert res[2] == 5  # Dedup thresh

    # --- Auto Thresholds ---

    def test_on_filters_changed_wrapper(self, app_ui, tmp_path):
        """Test on_filters_changed_wrapper uses ApplicationState data."""
        mock_ret = {"filter_status_text": "OK", "results_gallery": {"value": []}}

        with patch("ui.app_ui.on_filters_changed", return_value=mock_ret):
            out_dir = str(tmp_path / "analysis_out")
            (tmp_path / "analysis_out").mkdir()
            state = ApplicationState(all_frames_data=[{"f": 1}], analysis_output_dir=out_dir)
            slider_vals = [0.0, 0.0, 0.0]
            app_ui.components["metric_sliders"] = {}

            status, gallery_update = app_ui.on_filters_changed_wrapper(
                state, "Kept", False, 0.6, False, 5, "pHash", *slider_vals
            )

            assert str(status) == "OK"
            assert gallery_update["value"] == []

    def test_on_auto_set_thresholds(self, app_ui):
        app_ui.components["metric_auto_threshold_cbs"] = {"quality_score": MagicMock()}

        with patch("ui.app_ui.auto_set_thresholds") as mock_auto:
            mock_auto.return_value = {"slider_quality_score_min": gr.update(value=50)}

            updates = app_ui.on_auto_set_thresholds({}, 50, True)  # True for checkbox

            assert updates[0]["value"] == 50

    # --- Integration-ish Wrapper Tests ---

    @patch("ui.handlers.pipeline_handlers.execute_session_load")
    def test_run_session_load_wrapper(self, mock_load, app_ui, app_state, tmp_path):
        mock_load.return_value = {
            "run_config": {"source_path": "test.mp4"},
            "session_path": "/session",
            "scenes": [],
            "metadata_exists": True,
        }

        # Create a dummy session directory for the wrapper to find
        session_path = str(tmp_path / "session")
        (tmp_path / "session").mkdir()  # Ensure it exists
        gen = app_ui.pipeline_handler.run_session_load_wrapper(session_path, app_state)

        # First yield is status update
        next(gen)
        # Second yield is UI updates
        updates = next(gen)

        assert updates[app_ui.components["source_input"]]["value"] == "test.mp4"
        assert updates[app_ui.components["unified_status"]] == "Session Loaded."
