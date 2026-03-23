from unittest.mock import MagicMock, patch


@patch("core.pipelines.ExtractionPipeline")
@patch("core.pipelines.PreAnalysisPipeline")
@patch("core.pipelines.AnalysisPipeline")
def test_execute_extraction_smoke(mock_analysis, mock_pre_analysis, mock_extraction, mock_ui_state):
    """
    End-to-end mock test for execute_extraction to ensure manager wiring succeeds.
    """
    from core.pipelines import execute_extraction

    mock_extraction.return_value.run.return_value = {"done": True, "output_dir": "/tmp/out"}
    mock_pre_analysis.return_value.run.return_value = [MagicMock()]
    mock_analysis.return_value.run_full_analysis.return_value = {"done": True, "output_dir": "/tmp/out"}

    import threading
    from queue import Queue

    from core.config import Config
    from core.logger import AppLogger
    from core.models import AnalysisParameters

    mock_ui_state["video_path"] = "/fake/video.mp4"
    mock_ui_state["output_folder"] = "/tmp/out"

    params = AnalysisParameters(**mock_ui_state)

    generator = execute_extraction(
        params,
        progress_queue=Queue(),
        cancel_event=threading.Event(),
        logger=MagicMock(spec=AppLogger),
        config=MagicMock(spec=Config),
    )

    results = list(generator)

    assert len(results) > 0
    final_result = results[-1]

    assert final_result.get("done") is True
