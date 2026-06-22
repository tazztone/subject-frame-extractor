import threading
from unittest.mock import MagicMock, patch

import pytest

from core.context import AnalysisContext
from core.events import ExtractionEvent, PreAnalysisEvent
from core.models import (
    AnalysisResult,
    ExtractionResult,
    PipelineFailure,
    PreAnalysisResult,
    PropagationResult,
)
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

    @pytest.fixture
    def context(self):
        return AnalysisContext(
            config=MagicMock(),
            logger=MagicMock(),
            progress_queue=MagicMock(),
            cancel_event=threading.Event(),
            thumbnail_manager=MagicMock(),
            model_registry=MagicMock(),
            cuda_available=True,
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
        context,
    ):
        """Test full chain for video: Pre -> Prop -> Ana."""
        mock_init_models.return_value = {"models": "loaded"}

        # Pre-analysis yields something then sets done=True
        mock_execute_pre_analysis.side_effect = _make_gen(
            PreAnalysisResult(
                unified_log="Pre done", scenes=[{"shot_id": 1}], output_dir="/tmp/out", video_path="video.mp4"
            )
        )

        # Propagation yields results
        mock_execute_propagation.side_effect = _make_gen(
            PropagationResult(unified_log="Prop done", output_dir="/tmp/out")
        )

        # Analysis yields results
        mock_execute_analysis.side_effect = _make_gen(
            AnalysisResult(unified_log="Ana done", output_dir="/tmp/out", metadata_path="/tmp/out/metadata.db")
        )

        gen = execute_analysis_orchestrator(
            event=mock_pre_analysis_event,
            context=context,
        )

        results = list(gen)

        assert any(isinstance(r, PropagationResult) and r.unified_log == "Prop done" for r in results)
        assert any(isinstance(r, AnalysisResult) and r.unified_log == "Ana done" for r in results)

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
        context,
    ):
        """Test folder mode skips propagation."""
        mock_pre_analysis_event.video_path = ""  # Folder mode
        mock_init_models.return_value = {}

        mock_execute_pre_analysis.side_effect = _make_gen(
            PreAnalysisResult(unified_log="Pre done", scenes=[], output_dir="/tmp/out", video_path="")
        )
        mock_execute_analysis.side_effect = _make_gen(AnalysisResult(unified_log="Ana done", output_dir="/tmp/out"))

        gen = execute_analysis_orchestrator(
            event=mock_pre_analysis_event,
            context=context,
        )

        results = list(gen)

        assert any(isinstance(r, PropagationResult) and "Skipped for Folder" in r.unified_log for r in results)
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
        context,
    ):
        """Chain stops if Pre-Analysis does not complete."""
        mock_init_models.return_value = {}

        # Yields failure early
        mock_execute_pre_analysis.side_effect = _make_gen(
            PipelineFailure(unified_log="Failed early", status_message="Error", error_message="failed")
        )

        gen = execute_analysis_orchestrator(
            event=mock_pre_analysis_event,
            context=context,
        )

        results = list(gen)

        assert len(results) == 1
        assert isinstance(results[0], PipelineFailure)
        assert results[0].unified_log == "Failed early"
        mock_execute_propagation.assert_not_called()

    @patch("core.pipelines.execute_extraction")
    @patch("core.pipelines.execute_analysis_orchestrator")
    def test_execute_full_pipeline_success(
        self,
        mock_orchestrator,
        mock_execute_extraction,
        mock_extraction_event,
        context,
    ):
        """Test full pipeline chain: Extraction -> Analysis Orchestrator."""
        mock_execute_extraction.side_effect = _make_gen(
            ExtractionResult(unified_log="Ext done", video_path="v.mp4", output_dir="/tmp/out")
        )
        mock_orchestrator.side_effect = _make_gen(
            AnalysisResult(unified_log="Orchestrator working", output_dir="/tmp/out")
        )

        gen = execute_full_pipeline(
            event=mock_extraction_event,
            context=context,
        )

        results = list(gen)

        assert any(
            isinstance(r, PropagationResult) and r.unified_log == "Moving to Analysis stages..." for r in results
        )
        assert any(isinstance(r, AnalysisResult) and r.unified_log == "Orchestrator working" for r in results)
        mock_orchestrator.assert_called_once()

        # Verify event passed to orchestrator
        call_args_list = mock_orchestrator.call_args[0]
        passed_event = call_args_list[0]
        assert passed_event.output_folder == "/tmp/out"
        assert passed_event.video_path == "v.mp4"

    @patch("core.pipelines.execute_extraction")
    @patch("core.pipelines.execute_analysis_orchestrator")
    def test_execute_full_pipeline_extraction_failure(
        self,
        mock_orchestrator,
        mock_execute_extraction,
        mock_extraction_event,
        context,
    ):
        """Chain stops if Extraction fails."""
        mock_execute_extraction.side_effect = _make_gen(
            PipelineFailure(unified_log="Extraction failed", status_message="Error", error_message="failed")
        )

        gen = execute_full_pipeline(
            event=mock_extraction_event,
            context=context,
        )

        results = list(gen)

        assert len(results) == 1
        assert isinstance(results[0], PipelineFailure)
        assert results[0].unified_log == "Extraction failed"
        mock_orchestrator.assert_not_called()
