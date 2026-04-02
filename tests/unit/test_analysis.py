import threading
from queue import Queue
from unittest.mock import MagicMock, patch

import numpy as np

from core.managers.analysis import AnalysisPipeline, PreAnalysisPipeline
from core.models import AnalysisParameters, Scene


@patch("core.managers.analysis.initialize_analysis_models")
@patch("core.managers.analysis.SubjectMasker")
@patch("core.managers.analysis.save_scene_seeds")
def test_pre_analysis_pipeline(mock_save, mock_masker, mock_init, mock_logger, mock_config, tmp_path):
    mock_init.return_value = {
        "device": "cpu",
        "face_analyzer": None,
        "ref_emb": None,
        "face_landmarker": None,
        "person_detector": None,
    }
    params = AnalysisParameters(source_path="video.mp4", output_folder=str(tmp_path))
    tm = MagicMock()
    reg = MagicMock()
    pipeline = PreAnalysisPipeline(mock_config, mock_logger, params, Queue(), threading.Event(), tm, reg)
    scenes = [Scene(shot_id=0, start_frame=0, end_frame=10)]

    mock_masker_instance = mock_masker.return_value
    mock_masker_instance.get_seed_for_frame.return_value = ([0, 0, 10, 10], {"score": 1.0})
    mock_masker_instance._create_frame_map.return_value = {0: "frame_000000.png"}
    mock_masker_instance.draw_bbox.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    # PreAnalysisPipeline.run expects thumbnails in previews_dir
    tm.get.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    res = pipeline.run(scenes)
    assert len(res) == 1
    assert mock_save.called


@patch("core.managers.analysis.initialize_analysis_models")
@patch("core.managers.analysis.SubjectMasker")
@patch("core.managers.analysis.Database")
@patch("core.managers.analysis.create_frame_map")
def test_analysis_pipeline_full(mock_cfm, mock_db, mock_masker, mock_init, mock_logger, mock_config, tmp_path):
    mock_init.return_value = {
        "device": "cpu",
        "face_analyzer": None,
        "ref_emb": None,
        "face_landmarker": None,
        "person_detector": None,
    }
    params = AnalysisParameters(source_path="video.mp4", video_path="video.mp4", output_folder=str(tmp_path))
    tm = MagicMock()
    reg = MagicMock()
    pipeline = AnalysisPipeline(mock_config, mock_logger, params, Queue(), threading.Event(), tm, reg)
    scenes = [Scene(shot_id=0, start_frame=0, end_frame=10)]
    res = pipeline.run_full_analysis(scenes)
    assert res["done"] is True
    assert (tmp_path / "progress.json").exists()


@patch("core.managers.analysis.initialize_analysis_models")
@patch("core.managers.analysis.Database")
def test_analysis_pipeline_only(mock_db, mock_init, mock_logger, mock_config, tmp_path):
    mock_init.return_value = {
        "device": "cpu",
        "face_analyzer": None,
        "ref_emb": None,
        "face_landmarker": None,
        "person_detector": None,
    }
    params = AnalysisParameters(source_path="video.mp4", output_folder=str(tmp_path))
    tm = MagicMock()
    reg = MagicMock()
    pipeline = AnalysisPipeline(mock_config, mock_logger, params, Queue(), threading.Event(), tm, reg)
    with patch.object(pipeline, "_run_analysis_loop"):
        res = pipeline.run_analysis_only([])
        assert res["done"] is True
