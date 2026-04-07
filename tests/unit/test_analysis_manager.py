import threading
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.managers.analysis import AnalysisPipeline, PreAnalysisPipeline
from core.models import AnalysisParameters, Scene


@pytest.fixture
def mock_deps():
    config = MagicMock()
    config.seeding_face_similarity_threshold = 0.5
    config.retry_max_attempts = 3
    config.retry_backoff_seconds = 1
    config.analysis_default_workers = 4
    config.analysis_default_batch_size = 2
    logger = MagicMock()
    cancel_event = threading.Event()
    progress_queue = Queue()
    thumbnail_manager = MagicMock()
    model_registry = MagicMock()
    return {
        "config": config,
        "logger": logger,
        "cancel_event": cancel_event,
        "progress_queue": progress_queue,
        "thumbnail_manager": thumbnail_manager,
        "model_registry": model_registry,
    }


@pytest.fixture
def analysis_params(tmp_path):
    p = AnalysisParameters(source_path="test.mp4")
    p.output_folder = str(tmp_path)
    p.video_path = "test.mp4"
    return p


@patch("core.managers.analysis.initialize_analysis_models")
@patch("core.scene_utils.helpers.save_scene_seeds")
@patch("core.managers.analysis.Database")
@patch("core.managers.analysis.run_operators")
def test_run_analysis_only_success(
    mock_run_ops, mock_db, mock_save_seeds, mock_init_models, mock_deps, analysis_params
):
    mock_init_models.return_value = {
        "device": "cpu",
        "face_analyzer": MagicMock(),
        "ref_emb": None,
        "face_landmarker": None,
        "subject_detector": None,
    }
    mock_run_ops.return_value = {"quality": MagicMock(success=True, metrics={"quality_score": 80})}

    pipeline = AnalysisPipeline(
        config=mock_deps["config"],
        logger=mock_deps["logger"],
        params=analysis_params,
        progress_queue=mock_deps["progress_queue"],
        cancel_event=mock_deps["cancel_event"],
        thumbnail_manager=mock_deps["thumbnail_manager"],
        model_registry=mock_deps["model_registry"],
        database=mock_db,
    )
    pipeline.thumb_dir.mkdir(parents=True, exist_ok=True)
    pipeline.frame_map = {10: "f.webp"}

    scenes = [Scene(shot_id=1, start_frame=1, end_frame=20, best_frame=10)]
    result = pipeline.run_analysis_only(scenes)
    assert result["done"] is True


@patch("core.managers.analysis.initialize_analysis_models")
@patch("core.managers.analysis.SubjectMasker")
@patch("core.scene_utils.save_scene_seeds")
def test_pre_analysis_pipeline_run(mock_save_seeds, mock_masker_cls, mock_init_models, mock_deps, analysis_params):
    pipeline = PreAnalysisPipeline(
        config=mock_deps["config"],
        logger=mock_deps["logger"],
        params=analysis_params,
        progress_queue=mock_deps["progress_queue"],
        cancel_event=mock_deps["cancel_event"],
        thumbnail_manager=mock_deps["thumbnail_manager"],
        model_registry=mock_deps["model_registry"],
    )
    scenes = [Scene(shot_id=1, start_frame=0, end_frame=10)]
    # Mocking SubjectMasker return values
    instance = mock_masker_cls.return_value
    instance.get_seed_for_frame.return_value = ([0, 0, 10, 10], {"confidence": 0.9})
    instance._create_frame_map.return_value = {0: "frame_000000.png"}
    instance.draw_bbox.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_deps["thumbnail_manager"].get.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

    result = pipeline.run(scenes)
    assert len(result) == 1
    assert mock_save_seeds.called


@patch("core.managers.analysis.cv2.imread")
def test_process_reference_face_logic(mock_imread, mock_deps, analysis_params):
    pipeline = AnalysisPipeline(
        config=mock_deps["config"],
        logger=mock_deps["logger"],
        params=analysis_params,
        progress_queue=mock_deps["progress_queue"],
        cancel_event=mock_deps["cancel_event"],
        thumbnail_manager=mock_deps["thumbnail_manager"],
        model_registry=mock_deps["model_registry"],
        database=MagicMock(),
    )
    ref_img = Path(analysis_params.output_folder) / "ref.jpg"
    ref_img.touch()
    analysis_params.face_ref_img_path = str(ref_img)
    mock_imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    pipeline.face_analyzer = MagicMock()
    mock_face = MagicMock()
    mock_face.normed_embedding = np.ones(128)
    mock_face.det_score = 0.9
    pipeline.face_analyzer.get.return_value = [mock_face]
    pipeline._process_reference_face()
    assert pipeline.reference_embedding is not None


def test_analysis_loop_batch_error(mock_deps, analysis_params):
    pipeline = AnalysisPipeline(
        config=mock_deps["config"],
        logger=mock_deps["logger"],
        params=analysis_params,
        progress_queue=mock_deps["progress_queue"],
        cancel_event=mock_deps["cancel_event"],
        thumbnail_manager=mock_deps["thumbnail_manager"],
        model_registry=mock_deps["model_registry"],
        database=MagicMock(),
    )
    scenes = [Scene(shot_id=1, start_frame=1, end_frame=10)]
    with patch.object(pipeline, "_process_batch", side_effect=Exception("Batch Fail")):
        pipeline._run_analysis_loop(scenes, {}, tracker=None)
        assert mock_deps["logger"].error.called
