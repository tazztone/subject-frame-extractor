import threading
from queue import Queue
from unittest.mock import MagicMock, patch

from core.events import ExtractionEvent, PreAnalysisEvent, PropagationEvent
from core.models import Scene
from core.pipelines import execute_analysis, execute_extraction, execute_pre_analysis, execute_propagation


@patch("core.pipelines.ExtractionPipeline")
def test_execute_extraction(mock_pipeline_cls, mock_config, mock_logger):
    mock_pipeline = mock_pipeline_cls.return_value
    mock_pipeline.run.return_value = {"done": True, "output_dir": "/tmp/out", "video_path": "test.mp4"}

    event = ExtractionEvent(
        source_path="test.mp4",
        method="all",
        interval=1.0,
        nth_frame=1,
        max_resolution="720",
        thumb_megapixels=0.5,
        scene_detect=False,
    )

    results = list(execute_extraction(event, Queue(), threading.Event(), mock_logger, mock_config))
    assert results[-1]["done"] is True
    assert results[-1]["extracted_frames_dir_state"] == "/tmp/out"


@patch("core.pipelines.PreAnalysisPipeline")
@patch("core.pipelines._load_scenes", return_value=[])
def test_execute_pre_analysis(mock_load, mock_pipeline_cls, mock_config, mock_logger, tmp_path):
    mock_pipeline = mock_pipeline_cls.return_value
    mock_pipeline.run.return_value = []

    event = PreAnalysisEvent(
        output_folder=str(tmp_path),
        video_path="test.mp4",
        face_model_name="yolov8n-face",
        tracker_model_name="botsort",
        best_frame_strategy="middle",
        min_mask_area_pct=0.1,
        sharpness_base_scale=1.0,
        edge_strength_base_scale=1.0,
        pre_sample_nth=1,
        primary_seed_strategy="first",
    )

    # We need to mock gr.update for gradio
    with patch("gradio.update", return_value=None):
        results = list(
            execute_pre_analysis(event, Queue(), threading.Event(), mock_logger, mock_config, MagicMock(), False)
        )
        assert results[-1]["done"] is True


@patch("core.pipelines.AnalysisPipeline")
@patch("core.pipelines._load_analysis_scenes")
@patch("core.pipelines.VideoManager.get_video_info", return_value={"fps": 30, "frame_count": 100})
def test_execute_propagation(mock_vinfo, mock_load, mock_pipeline_cls, mock_config, mock_logger, tmp_path):
    mock_pipeline = mock_pipeline_cls.return_value
    mock_pipeline.run_full_analysis.return_value = {"done": True, "output_dir": str(tmp_path)}
    mock_load.return_value = [Scene(shot_id=1, start_frame=0, end_frame=10)]

    mock_params = PreAnalysisEvent(
        output_folder=str(tmp_path),
        video_path="test.mp4",
        face_model_name="yolov8n-face",
        tracker_model_name="botsort",
        best_frame_strategy="middle",
        min_mask_area_pct=0.1,
        sharpness_base_scale=1.0,
        edge_strength_base_scale=1.0,
        pre_sample_nth=1,
        primary_seed_strategy="first",
    )
    event = PropagationEvent(scenes=[], analysis_params=mock_params, output_folder=str(tmp_path), video_path="test.mp4")

    results = list(execute_propagation(event, Queue(), threading.Event(), mock_logger, mock_config, MagicMock(), False))
    assert results[-1]["done"] is True


@patch("core.pipelines.AnalysisPipeline")
@patch("core.pipelines._load_analysis_scenes")
def test_execute_analysis(mock_load, mock_pipeline_cls, mock_config, mock_logger, tmp_path):
    mock_pipeline = mock_pipeline_cls.return_value
    mock_pipeline.run_analysis_only.return_value = {"done": True, "output_dir": str(tmp_path)}
    mock_load.return_value = [MagicMock()]

    mock_params = PreAnalysisEvent(
        output_folder=str(tmp_path),
        video_path="test.mp4",
        face_model_name="yolov8n-face",
        tracker_model_name="botsort",
        best_frame_strategy="middle",
        min_mask_area_pct=0.1,
        sharpness_base_scale=1.0,
        edge_strength_base_scale=1.0,
        pre_sample_nth=1,
        primary_seed_strategy="first",
    )
    event = PropagationEvent(scenes=[], analysis_params=mock_params, output_folder=str(tmp_path), video_path="test.mp4")

    results = list(execute_analysis(event, Queue(), threading.Event(), mock_logger, mock_config, MagicMock(), False))
    assert results[-1]["done"] is True
