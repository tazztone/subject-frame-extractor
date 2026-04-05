from queue import Queue
from unittest.mock import MagicMock, patch

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
        self,
        mock_config,
        mock_logger,
        mock_queue,
        mock_cancel_event,
        mock_thumbnail_manager,
        mock_model_registry,
        mock_database,
    ):
        # We need to mock torch.cuda.is_available inside AppUI init
        with patch("torch.cuda.is_available", return_value=False, create=True):
            ui = AppUI(
                mock_config,
                mock_logger,
                mock_queue,
                mock_cancel_event,
                mock_thumbnail_manager,
                mock_model_registry,
                mock_database,
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
            "compute_face_sim": MagicMock(),
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
            "export_button": MagicMock(),
            "dry_run_button": MagicMock(),
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
        with patch.object(app_ui, "_run_pipeline") as mock_run:
            mock_run.return_value = iter([])
            args = ["vid.mp4", None, "interval", 1.0, 1, "1080", 0.5, True]
            list(app_ui.pipeline_handler.run_extraction_wrapper(app_state, *args))
            mock_run.assert_called_once()
            event = mock_run.call_args[0][1]
            assert event.source_path == "vid.mp4"

    def test_run_pre_analysis_wrapper(self, app_ui, app_state, tmp_path):
        out_dir = str(tmp_path / "out")
        with patch.object(app_ui, "_run_pipeline") as mock_run:
            mock_run.return_value = iter([])
            args = [
                False,  # resume
                "",  # face_ref_img_path
                None,  # face_ref_img_upload
                "buffalo_l",  # face_model_name
                True,  # enable_subject_mask
                "sam2",  # tracker_model_name
                "Largest",  # best_frame_strategy
                True,  # scene_detect
                "",  # text_prompt
                1.0,  # min_mask_area_pct
                2500.0,  # sharpness_base_scale
                100.0,  # edge_strength_base_scale
                True,  # pre_analysis_enabled
                1,  # pre_sample_nth
                "YOLO12l-Seg",  # subject_detector_model
                "person",  # subject_detector_class_name
                0.45,  # subject_detector_threshold
                "Automatic Detection",  # primary_seed_strategy
                True,  # compute_quality_score
                True,  # compute_sharpness
            ] + [True] * 12

            app_state.extracted_video_path = "vid.mp4"
            app_state.analysis_output_dir = out_dir

            list(app_ui.pipeline_handler.run_pre_analysis_wrapper(app_state, *args))
            mock_run.assert_called_once()

    # --- Success Callbacks ---

    def test_on_extraction_success(self, app_ui, app_state):
        res = {"extracted_video_path_state": "v.mp4", "extracted_frames_dir_state": "/frames"}
        updates = app_ui.pipeline_handler._on_extraction_success(res, app_state)
        new_state = updates[app_ui.components["application_state"]]
        assert new_state.extracted_video_path == "v.mp4"

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
        res = {"unified_log": "Success", "scenes": scenes, "output_dir": str(tmp_path / "out"), "video_path": "vid.mp4"}

        with patch("ui.handlers.pipeline_handlers.get_scene_status_text", return_value=("Status", "Button")):
            updates = app_ui.pipeline_handler._on_pre_analysis_success(res, app_state)

        new_state = updates[app_ui.components["application_state"]]
        assert new_state.scenes == scenes
        assert updates[app_ui.components["unified_log"]] == "Success"

    # --- History/Undo ---

    def test_push_history(self, app_state):
        scenes = [{"id": 1}]
        app_state.push_history(scenes)
        assert len(app_state.scene_history) == 1

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

    # --- Integration-ish Wrapper Tests ---

    def test_run_session_load_wrapper(self, app_ui, app_state, tmp_path):
        """Verify run_session_load_wrapper uses _run_pipeline orchestrator."""
        with patch.object(app_ui, "_run_pipeline") as mock_run:
            mock_run.return_value = iter([])
            session_path = str(tmp_path / "session")
            (tmp_path / "session").mkdir()

            list(app_ui.pipeline_handler.run_session_load_wrapper(session_path, app_state))

            mock_run.assert_called_once()
            # Verify the adapter is used as the first arg to _run_pipeline
            assert mock_run.call_args[0][0] == app_ui.pipeline_handler._execute_session_load_adapter
