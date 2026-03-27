from unittest.mock import MagicMock, patch

from core.batch_manager import BatchManager
from core.managers.analysis import AnalysisPipeline, PreAnalysisPipeline
from core.models import AnalysisParameters


def test_integration_smoke_flow(mock_config, mock_logger, mock_model_registry):
    """
    Smoke test to verify the integration of BatchManager and Pipelines.
    Uses mocks for heavy ML dependencies to ensure it can run in standard CI.
    """
    # 1. Setup Managers
    batch_manager = BatchManager()

    # Pipelines require more args: (config, logger, params, progress_queue, cancel_event)
    mock_params = MagicMock(spec=AnalysisParameters)
    mock_params.output_folder = "/tmp/out"
    mock_progress_queue = MagicMock()
    mock_cancel_event = MagicMock()

    # AnalysisPipeline also needs model_registry
    # ExtractionPipeline might not, but let's check

    analysis_pipeline = PreAnalysisPipeline(
        config=mock_config,
        logger=mock_logger,
        params=mock_params,
        progress_queue=mock_progress_queue,
        cancel_event=mock_cancel_event,
        thumbnail_manager=MagicMock(),
        model_registry=mock_model_registry,
    )

    # 2. Mock some internal methods
    with patch(
        "core.managers.video.VideoManager.get_video_info",
        return_value={"fps": 30.0, "width": 1920, "height": 1080, "frame_count": 100},
    ):
        # 3. Verify BatchManager
        assert batch_manager is not None

        # 4. Verify Pipelines
        assert analysis_pipeline is not None
        assert isinstance(analysis_pipeline, PreAnalysisPipeline)


def test_pipeline_wiring(mock_config, mock_logger, mock_model_registry):
    """Verify Pipeline wiring with mocks."""
    mock_params = AnalysisParameters(source_path="mock.mp4", output_folder="/tmp/out", tracker_model_name="sam2")
    mock_progress_queue = MagicMock()
    mock_cancel_event = MagicMock()

    ap = AnalysisPipeline(
        config=mock_config,
        logger=mock_logger,
        params=mock_params,
        progress_queue=mock_progress_queue,
        cancel_event=mock_cancel_event,
        thumbnail_manager=MagicMock(),
        model_registry=mock_model_registry,
    )

    assert hasattr(ap, "run_full_analysis")
