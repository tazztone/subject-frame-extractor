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
    from core.events import ExtractionEvent
    from core.logger import AppLogger

    event = ExtractionEvent(
        source_path="/fake/video.mp4",
        method="every_nth_frame",
        interval=1.0,
        nth_frame=1,
        max_resolution="1080",
        thumb_megapixels=0.5,
        scene_detect=True,
        output_folder="/tmp/out",
    )

    from core.context import AnalysisContext

    context = AnalysisContext(
        config=MagicMock(spec=Config),
        logger=MagicMock(spec=AppLogger),
        progress_queue=Queue(),
        cancel_event=threading.Event(),
        thumbnail_manager=MagicMock(),
        model_registry=MagicMock(),
        cuda_available=False,
    )

    generator = execute_extraction(event, context)

    results = list(generator)

    assert len(results) > 0
    final_result = results[-1]

    assert final_result.done is True
