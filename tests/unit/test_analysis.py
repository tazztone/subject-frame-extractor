import threading
from queue import Queue
from unittest.mock import MagicMock, patch

import numpy as np

from core.managers.analysis import AnalysisPipeline, PreAnalysisPipeline
from core.models import AnalysisParameters, Scene


@patch("core.managers.analysis.initialize_analysis_models")
@patch("core.managers.analysis.SubjectMasker")
@patch("core.scene_utils.save_scene_seeds")
def test_pre_analysis_pipeline(mock_save, mock_masker, mock_init, mock_logger, mock_config_simple, tmp_path):
    mock_init.return_value = {
        "device": "cpu",
        "face_analyzer": None,
        "ref_emb": None,
        "face_landmarker": None,
        "subject_detector": None,
    }
    params = AnalysisParameters(source_path="video.mp4", output_folder=str(tmp_path))
    tm = MagicMock()
    reg = MagicMock()
    pipeline = PreAnalysisPipeline(
        config=mock_config_simple,
        logger=mock_logger,
        params=params,
        progress_queue=Queue(),
        cancel_event=threading.Event(),
        thumbnail_manager=tm,
        model_registry=reg,
    )
    scenes = [Scene(shot_id=0, start_frame=0, end_frame=10)]
    mock_masker_instance = mock_masker.return_value
    mock_masker_instance.get_seed_for_frame.return_value = ([0, 0, 10, 10], {"score": 1.0})
    mock_masker_instance._create_frame_map.return_value = {0: "frame_000000.png"}
    mock_masker_instance.draw_bbox.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    tm.get.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    res = pipeline.run(scenes)
    assert len(res) == 1
    assert mock_save.called


@patch("core.managers.analysis.initialize_analysis_models")
@patch("core.managers.analysis.SubjectMasker")
@patch("core.managers.analysis.Database")
@patch("core.managers.analysis.create_frame_map")
def test_analysis_pipeline_full(mock_cfm, mock_db, mock_masker, mock_init, mock_logger, mock_config_simple, tmp_path):
    mock_init.return_value = {
        "device": "cpu",
        "face_analyzer": None,
        "ref_emb": None,
        "face_landmarker": None,
        "subject_detector": None,
    }
    params = AnalysisParameters(source_path="video.mp4", output_folder=str(tmp_path))
    params.video_path = "video.mp4"
    tm = MagicMock()
    reg = MagicMock()
    pipeline = AnalysisPipeline(
        config=mock_config_simple,
        logger=mock_logger,
        params=params,
        progress_queue=Queue(),
        cancel_event=threading.Event(),
        thumbnail_manager=tm,
        model_registry=reg,
        database=mock_db,
    )
    assert pipeline.db is not None
