from unittest.mock import MagicMock, patch

import pytest

from core.events import ExtractionEvent, PreAnalysisEvent
from core.pipelines import execute_analysis_orchestrator, execute_full_pipeline


def _make_gen(*results):
    """Helper to create a generator function that yields results."""

    def _gen(*args, **kwargs):
        for res in results:
            yield res

    return _gen


class TestPipelinesOrchestrator:
    @pytest.fixture
    def mock_extraction_event(self):
        return ExtractionEvent(
            source_path="video.mp4",
            output_folder="/tmp/out",
            method="every_nth_frame",
            interval=1.0,
            nth_frame=1,
            max_resolution="720",
            thumb_megapixels=0.5,
            scene_detect=True,
        )

    @pytest.fixture
    def mock_pre_analysis_event(self):
        return PreAnalysisEvent(
            output_folder="/tmp/out",
            video_path="video.mp4",
            face_ref_img_path="",
            primary_seed_strategy="Automatic Detection",
            resume=False,
        )

    @patch("core.pipelines.initialize_analysis_models")
    @patch("core.pipelines.execute_pre_analysis")
    @patch("core.pipelines.execute_propagation")
    @patch("core.pipelines.execute_analysis")
    def test_execute_analysis_orchestrator_video_full_chain(
        self,
        mock_execute_analysis,
        mock_execute_propagation,
        mock_execute_pre_analysis,
        mock_init_models,
        mock_pre_analysis_event,
        mock_progress_queue,
        mock_cancel_event,
        mock_logger,
        mock_config,
        mock_thumbnail_manager,
        mock_model_registry,
    ):
        """Test full chain for video: Pre -> Prop -> Ana."""
        mock_init_models.return_value = {"models": "loaded"}

        # Pre-analysis yields something then sets done=True
        mock_execute_pre_analysis.side_effect = _make_gen(
            {"unified_log": "Pre log", "done": False},
            {"unified_log": "Pre done", "done": True, "scenes": [{"shot_id": 1}]},
        )

        # Propagation yields results
        mock_execute_propagation.side_effect = _make_gen(
            {"unified_log": "Prop log", "done": False}, {"unified_log": "Prop done", "done": True}
        )

        # Analysis yields results
        mock_execute_analysis.side_effect = _make_gen(
            {"unified_log": "Ana log", "done": False}, {"unified_log": "Ana done", "done": True}
        )

        gen = execute_analysis_orchestrator(
            event=mock_pre_analysis_event,
            progress_queue=mock_progress_queue,
            cancel_event=mock_cancel_event,
            logger=mock_logger,
            config=mock_config,
            thumbnail_manager=mock_thumbnail_manager,
            cuda_available=True,
            model_registry=mock_model_registry,
            database=MagicMock(),
        )

        results = list(gen)

        assert any(r.get("unified_log") == "Pre done" for r in results)
        assert any(r.get("unified_log") == "Prop done" for r in results)
        assert any(r.get("unified_log") == "Ana done" for r in results)

        mock_execute_pre_analysis.assert_called_once()
        mock_execute_propagation.assert_called_once()
        mock_execute_analysis.assert_called_once()

    @patch("core.pipelines.initialize_analysis_models")
    @patch("core.pipelines.execute_pre_analysis")
    @patch("core.pipelines.execute_propagation")
    @patch("core.pipelines.execute_analysis")
    def test_execute_analysis_orchestrator_folder_mode(
        self,
        mock_execute_analysis,
        mock_execute_propagation,
        mock_execute_pre_analysis,
        mock_init_models,
        mock_pre_analysis_event,
        mock_progress_queue,
        mock_cancel_event,
        mock_logger,
        mock_config,
        mock_thumbnail_manager,
        mock_model_registry,
    ):
        """Test folder mode skips propagation."""
        mock_pre_analysis_event.video_path = ""  # Folder mode
        mock_init_models.return_value = {}

        mock_execute_pre_analysis.side_effect = _make_gen({"done": True, "scenes": []})
        mock_execute_analysis.side_effect = _make_gen({"done": True})

        gen = execute_analysis_orchestrator(
            event=mock_pre_analysis_event,
            progress_queue=mock_progress_queue,
            cancel_event=mock_cancel_event,
            logger=mock_logger,
            config=mock_config,
            thumbnail_manager=mock_thumbnail_manager,
            model_registry=mock_model_registry,
            database=MagicMock(),
            cuda_available=True,
        )

        results = list(gen)

        assert any("Skipped for Folder" in str(r.get("unified_log")) for r in results)
        mock_execute_propagation.assert_not_called()
        mock_execute_analysis.assert_called_once()

    @patch("core.pipelines.initialize_analysis_models")
    @patch("core.pipelines.execute_pre_analysis")
    @patch("core.pipelines.execute_propagation")
    def test_execute_analysis_orchestrator_pre_analysis_failure(
        self,
        mock_execute_propagation,
        mock_execute_pre_analysis,
        mock_init_models,
        mock_pre_analysis_event,
        mock_progress_queue,
        mock_cancel_event,
        mock_logger,
        mock_config,
    ):
        """Chain stops if Pre-Analysis does not complete."""
        mock_init_models.return_value = {}

        # Yields log but never yields done=True
        mock_execute_pre_analysis.side_effect = _make_gen({"unified_log": "Failed early", "done": False})

        gen = execute_analysis_orchestrator(
            event=mock_pre_analysis_event,
            progress_queue=mock_progress_queue,
            cancel_event=mock_cancel_event,
            logger=mock_logger,
            config=mock_config,
            thumbnail_manager=MagicMock(),
            model_registry=MagicMock(),
            database=MagicMock(),
            cuda_available=True,
        )

        results = list(gen)

        assert len(results) == 1
        assert results[0]["unified_log"] == "Failed early"
        mock_execute_propagation.assert_not_called()

    @patch("core.pipelines.execute_extraction")
    @patch("core.pipelines.execute_analysis_orchestrator")
    def test_execute_full_pipeline_success(
        self,
        mock_orchestrator,
        mock_execute_extraction,
        mock_extraction_event,
        mock_progress_queue,
        mock_cancel_event,
        mock_logger,
        mock_config,
        mock_thumbnail_manager,
    ):
        """Test full pipeline chain: Extraction -> Analysis Orchestrator."""
        mock_execute_extraction.side_effect = _make_gen(
            {"done": True, "extracted_frames_dir_state": "/tmp/out", "extracted_video_path_state": "v.mp4"}
        )
        mock_orchestrator.side_effect = _make_gen(
            {"unified_log": "Orchestrator working", "done": False}, {"done": True}
        )

        gen = execute_full_pipeline(
            event=mock_extraction_event,
            progress_queue=mock_progress_queue,
            cancel_event=mock_cancel_event,
            logger=mock_logger,
            config=mock_config,
            thumbnail_manager=mock_thumbnail_manager,
            model_registry=MagicMock(),
            database=MagicMock(),
            cuda_available=True,
        )

        results = list(gen)

        assert any(r.get("unified_log") == "Moving to Analysis stages..." for r in results)
        assert any(r.get("unified_log") == "Orchestrator working" for r in results)
        mock_orchestrator.assert_called_once()

        # Verify events passed to orchestrator
        # It's called with keyword arguments in core/pipelines.py
        passed_event = mock_orchestrator.call_args.kwargs["event"]
        assert passed_event.output_folder == "/tmp/out"
        assert passed_event.video_path == "v.mp4"

    @patch("core.pipelines.execute_extraction")
    @patch("core.pipelines.execute_analysis_orchestrator")
    def test_execute_full_pipeline_extraction_failure(
        self,
        mock_orchestrator,
        mock_execute_extraction,
        mock_extraction_event,
        mock_progress_queue,
        mock_cancel_event,
        mock_logger,
        mock_config,
    ):
        """Chain stops if Extraction fails."""
        mock_execute_extraction.side_effect = _make_gen({"done": False, "unified_log": "Extraction failed"})

        gen = execute_full_pipeline(
            event=mock_extraction_event,
            progress_queue=mock_progress_queue,
            cancel_event=mock_cancel_event,
            logger=mock_logger,
            config=mock_config,
            thumbnail_manager=MagicMock(),
            model_registry=MagicMock(),
            database=MagicMock(),
            cuda_available=True,
        )

        results = list(gen)

        assert len(results) == 1
        assert results[0]["unified_log"] == "Extraction failed"
        mock_orchestrator.assert_not_called()
