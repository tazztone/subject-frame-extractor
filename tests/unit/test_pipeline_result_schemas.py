import threading
from unittest.mock import MagicMock, patch

from core.context import AnalysisContext
from core.events import PropagationEvent
from core.pipelines import execute_analysis, execute_extraction


def test_execute_analysis_schema():
    """Verify that execute_analysis yields the expected metadata_path."""
    event = MagicMock(spec=PropagationEvent)
    event.scenes = []
    event.analysis_params = MagicMock()
    event.analysis_params.model_dump.return_value = {}

    context = AnalysisContext(
        config=MagicMock(),
        logger=MagicMock(),
        progress_queue=MagicMock(),
        cancel_event=threading.Event(),
        thumbnail_manager=MagicMock(),
        model_registry=MagicMock(),
        cuda_available=True,
    )

    # Mock AnalysisParameters.from_ui
    with patch("core.pipelines.AnalysisParameters.from_ui") as mock_params_v:
        mock_p = MagicMock()
        mock_p.video_path = "video.mp4"
        mock_p.output_folder = "/tmp/out"
        mock_params_v.return_value = mock_p

        mock_scene = MagicMock()
        mock_scene.start_frame = 0
        mock_scene.end_frame = 10
        with patch("core.pipelines._load_analysis_scenes", return_value=[mock_scene]):
            with patch("core.pipelines.MediaSession.get_video_info"):
                with patch("core.pipelines.AnalysisPipeline") as mock_init:
                    mock_pipe = MagicMock()
                    mock_pipe.run_analysis_only.return_value = {"done": True, "output_dir": "/tmp/out"}
                    mock_init.return_value = mock_pipe

                    gen = execute_analysis(event, context)

                    results = list(gen)
                    success_yield = results[-1]

                    assert success_yield.done is True
                    assert success_yield.metadata_path.endswith("metadata.db")


def test_execute_extraction_schema():
    """Verify that execute_extraction yields the expected keys."""
    event = MagicMock()
    event.model_dump.return_value = {}

    context = AnalysisContext(
        config=MagicMock(),
        logger=MagicMock(),
        progress_queue=MagicMock(),
        cancel_event=threading.Event(),
        thumbnail_manager=MagicMock(),
        model_registry=MagicMock(),
        cuda_available=True,
    )

    with patch("core.pipelines._handle_extraction_uploads", return_value={}):
        with patch("core.pipelines.AnalysisParameters.from_ui") as mock_init_p:
            mock_p = MagicMock()
            mock_init_p.return_value = mock_p

            with patch("core.pipelines.ExtractionPipeline") as mock_pipe_class:
                mock_pipe = MagicMock()
                mock_pipe.run.return_value = {"done": True, "output_dir": "/tmp/out", "video_path": "vid.mp4"}
                mock_pipe_class.return_value = mock_pipe

                gen = execute_extraction(event, context)

                results = list(gen)
                success_yield = results[-1]

                assert success_yield.done is True
                assert success_yield.output_dir == "/tmp/out"
                assert success_yield.video_path == "vid.mp4"
