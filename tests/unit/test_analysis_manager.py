import json
import threading
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.managers.analysis import AnalysisPipeline, SceneStatus
from core.models import AnalysisParameters, Scene


@pytest.fixture
def mock_deps():
    config = MagicMock()
    config.analysis_default_workers = 4
    config.analysis_default_batch_size = 1
    return {
        "config": config,
        "logger": MagicMock(),
        "progress_queue": Queue(),
        "cancel_event": threading.Event(),
        "thumbnail_manager": MagicMock(),
        "model_registry": MagicMock(),
    }


@pytest.fixture
def analysis_params():
    return AnalysisParameters(
        output_folder="/tmp/out",
        video_path="test.mp4",
        face_model_name="buffalo_l",
        tracker_model_name="vit",
        best_frame_strategy="sharpness",
        min_mask_area_pct=0.1,
        sharpness_base_scale=1.0,
        edge_strength_base_scale=1.0,
        primary_seed_strategy="Find Prominent Person",
    )


def test_analysis_pipeline_initialization(mock_deps, analysis_params):
    pipeline = AnalysisPipeline(
        mock_deps["config"],
        mock_deps["logger"],
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
        mock_deps["thumbnail_manager"],
        mock_deps["model_registry"],
    )
    assert pipeline.params == analysis_params
    assert pipeline.output_dir == Path(analysis_params.output_folder)


def test_run_full_analysis_no_scenes(mock_deps, analysis_params, tmp_path):
    analysis_params.output_folder = str(tmp_path)
    pipeline = AnalysisPipeline(
        mock_deps["config"],
        mock_deps["logger"],
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
        mock_deps["thumbnail_manager"],
        mock_deps["model_registry"],
    )
    result = pipeline.run_full_analysis([])
    assert result["done"] is True
    assert result["output_dir"] == str(tmp_path)


@patch("core.managers.analysis.SubjectMasker")
@patch("core.managers.analysis.initialize_analysis_models")
def test_run_full_analysis_success(mock_init_models, mock_masker_cls, mock_deps, analysis_params, tmp_path):
    analysis_params.output_folder = str(tmp_path)
    analysis_params.video_path = "test.mp4"

    pipeline = AnalysisPipeline(
        mock_deps["config"],
        mock_deps["logger"],
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
        mock_deps["thumbnail_manager"],
        mock_deps["model_registry"],
    )
    pipeline.db = MagicMock()

    scenes = [Scene(shot_id=1, start_frame=0, end_frame=10)]
    mock_init_models.return_value = {"face_analyzer": None, "ref_emb": None, "face_landmarker": None, "device": "cpu"}

    result = pipeline.run_full_analysis(scenes)

    assert result["done"] is True
    assert mock_masker_cls.return_value.run_propagation.called


@patch("core.managers.analysis.run_operators")
@patch("core.managers.analysis.initialize_analysis_models")
def test_run_analysis_only_success(mock_init_models, mock_run_ops, mock_deps, analysis_params, tmp_path):
    analysis_params.output_folder = str(tmp_path)
    analysis_params.video_path = "test.mp4"
    analysis_params.compute_quality_score = True

    pipeline = AnalysisPipeline(
        mock_deps["config"],
        mock_deps["logger"],
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
        mock_deps["thumbnail_manager"],
        mock_deps["model_registry"],
    )
    pipeline.db = MagicMock()
    pipeline.thumb_dir = tmp_path / "thumbs"
    pipeline.thumb_dir.mkdir()
    (pipeline.thumb_dir / "frame_000001.webp").write_text("dummy")

    mock_init_models.return_value = {"face_analyzer": None, "ref_emb": None, "face_landmarker": None, "device": "cpu"}
    mock_run_ops.return_value = {"quality": MagicMock(success=True, metrics={"quality_score": 80})}

    # Need a frame_map in the pipeline
    pipeline.mask_metadata = {"frame_000001": {"mask_path": None}}

    # Mocking thumb_manager and cv2
    mock_deps["thumbnail_manager"].get.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

    with patch("core.managers.analysis.create_frame_map", return_value={1: "frame_000001.webp"}):
        scenes = [Scene(shot_id=1, start_frame=1, end_frame=1)]
        result = pipeline.run_analysis_only(scenes)

    if not result["done"]:
        print(f"DEBUG: result={result}")
    assert result["done"] is True
    assert mock_run_ops.called


@patch("core.managers.analysis.initialize_analysis_models")
def test_run_full_analysis_cancellation(mock_init_models, mock_deps, analysis_params, tmp_path):
    analysis_params.output_folder = str(tmp_path)
    pipeline = AnalysisPipeline(
        mock_deps["config"],
        mock_deps["logger"],
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
        mock_deps["thumbnail_manager"],
        mock_deps["model_registry"],
    )
    pipeline.db = MagicMock()
    mock_init_models.return_value = {"face_analyzer": None, "ref_emb": None, "face_landmarker": None, "device": "cpu"}

    with patch("core.managers.analysis.SubjectMasker"):
        # Cancellation check is after masker.run_propagation
        mock_deps["cancel_event"].set()

        scenes = [Scene(shot_id=1, start_frame=0, end_frame=10)]
        result = pipeline.run_full_analysis(scenes)

    assert result["done"] is False
    assert "cancelled" in result["log"].lower()


@patch("core.managers.analysis.initialize_analysis_models")
@patch("core.managers.analysis.SubjectMasker")
def test_error_propagation_in_analysis(mock_masker_cls, mock_init_models, mock_deps, analysis_params, tmp_path):
    analysis_params.output_folder = str(tmp_path)
    pipeline = AnalysisPipeline(
        mock_deps["config"],
        mock_deps["logger"],
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
        mock_deps["thumbnail_manager"],
        mock_deps["model_registry"],
    )
    pipeline.db = MagicMock()
    mock_init_models.return_value = {"face_analyzer": None, "ref_emb": None, "face_landmarker": None, "device": "cpu"}

    scenes = [Scene(shot_id=1, start_frame=0, end_frame=10)]
    mock_masker_cls.return_value.run_propagation.side_effect = RuntimeError("IO Error")

    result = pipeline.run_full_analysis(scenes)
    assert result["done"] is False
    assert "IO Error" in result["error"]


@patch("core.managers.analysis.initialize_analysis_models")
def test_run_image_folder_analysis(mock_init_models, mock_deps, analysis_params, tmp_path):
    analysis_params.video_path = ""  # Folder mode
    analysis_params.output_folder = str(tmp_path)

    pipeline = AnalysisPipeline(
        mock_deps["config"],
        mock_deps["logger"],
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
        mock_deps["thumbnail_manager"],
        mock_deps["model_registry"],
    )
    pipeline.db = MagicMock()
    mock_init_models.return_value = {"face_analyzer": None, "ref_emb": None, "face_landmarker": None, "device": "cpu"}

    # Mock _load_scenes to return a scene
    with (
        patch(
            "core.managers.analysis._load_scenes",
            return_value=[Scene(shot_id=1, start_frame=0, end_frame=0, status=SceneStatus.INCLUDED)],
        ),
        patch("core.managers.analysis.create_frame_map", return_value={0: "img1.jpg"}),
        patch.object(pipeline, "_run_analysis_loop") as mock_loop,
    ):
        result = pipeline.run_full_analysis([Scene(shot_id=1, start_frame=0, end_frame=0)])

    assert result["done"] is True
    assert mock_loop.called


@patch("core.managers.analysis.ThreadPoolExecutor")
@patch("core.managers.analysis.run_operators")
def test_run_analysis_loop(mock_run_ops, mock_executor_cls, mock_deps, analysis_params, tmp_path):
    pipeline = AnalysisPipeline(
        mock_deps["config"],
        mock_deps["logger"],
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
        mock_deps["thumbnail_manager"],
        mock_deps["model_registry"],
    )

    scenes = [Scene(shot_id=1, start_frame=0, end_frame=1)]
    metrics = {"quality": True}

    # Setup mock executor to immediately return results
    mock_executor = mock_executor_cls.return_value.__enter__.return_value
    mock_future = MagicMock()
    mock_future.result.return_value = 1
    mock_executor.submit.return_value = mock_future

    # Frame map setup
    with patch(
        "core.managers.analysis.create_frame_map", return_value={0: "frame_000000.webp", 1: "frame_000001.webp"}
    ):
        pipeline._run_analysis_loop(scenes, metrics)

    assert mock_executor.submit.called


@patch("core.managers.analysis.run_operators")
def test_process_single_frame(mock_run_ops, mock_deps, analysis_params):
    pipeline = AnalysisPipeline(
        mock_deps["config"],
        mock_deps["logger"],
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
        mock_deps["thumbnail_manager"],
        mock_deps["model_registry"],
    )
    pipeline.db = MagicMock()
    pipeline.mask_metadata = {"frame_000000": {"mask_path": "mask.png"}}
    pipeline.masks_dir = Path("/tmp/masks")

    # Mock face analyzer result
    mock_face = MagicMock()
    mock_face.det_score = 0.9
    mock_face.bbox = np.array([0, 0, 10, 10])
    pipeline.face_analyzer = MagicMock()
    pipeline.face_analyzer.get.return_value = [mock_face]

    # Mock thumbnail
    mock_deps["thumbnail_manager"].get.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    # Mock operator result
    mock_op_res = MagicMock()
    mock_op_res.success = True
    mock_op_res.metrics = {"face_sim": 0.8, "face_conf": 0.9}
    mock_op_res.data = {"phash": "abc"}
    mock_run_ops.return_value = {"face": mock_op_res}

    with (
        patch("core.managers.analysis.cv2.imread", return_value=np.zeros((100, 100), dtype=np.uint8)),
        patch("core.managers.analysis.cv2.resize", side_effect=lambda x, y, **kwargs: x),
    ):
        pipeline._process_single_frame(Path("frame_000000.webp"), {"quality": True})

    assert pipeline.db.insert_metadata.called
    args, _ = pipeline.db.insert_metadata.call_args
    meta = args[0]
    assert meta["filename"] == "frame_000000.webp"
    assert meta["face_sim"] == 0.8


def test_filter_completed_scenes(mock_deps, analysis_params):
    pipeline = AnalysisPipeline(
        mock_deps["config"],
        mock_deps["logger"],
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
        mock_deps["thumbnail_manager"],
        mock_deps["model_registry"],
    )

    scenes = [Scene(shot_id=1, start_frame=0, end_frame=10), Scene(shot_id=2, start_frame=11, end_frame=20)]
    progress_data = {"completed_scenes": [1]}

    filtered = pipeline._filter_completed_scenes(scenes, progress_data)
    assert len(filtered) == 1
    assert filtered[0].shot_id == 2


def test_save_progress_bulk(mock_deps, analysis_params, tmp_path):
    pipeline = AnalysisPipeline(
        mock_deps["config"],
        mock_deps["logger"],
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
        mock_deps["thumbnail_manager"],
        mock_deps["model_registry"],
    )

    progress_file = tmp_path / "progress.json"
    pipeline._save_progress_bulk([1], progress_file)

    assert progress_file.exists()
    with open(progress_file, "r") as f:
        data = json.load(f)
        assert 1 in data["completed_scenes"]


@patch("core.managers.analysis.initialize_analysis_models")
@patch("core.managers.analysis.SubjectMasker")
@patch("core.managers.analysis.save_scene_seeds")
def test_pre_analysis_pipeline_run(
    mock_save_seeds, mock_masker_cls, mock_init_models, mock_deps, analysis_params, tmp_path
):
    analysis_params.output_folder = str(tmp_path)
    from core.managers.analysis import PreAnalysisPipeline

    pipeline = PreAnalysisPipeline(
        mock_deps["config"],
        mock_deps["logger"],
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
        mock_deps["thumbnail_manager"],
        mock_deps["model_registry"],
    )

    mock_init_models.return_value = {"face_analyzer": None, "ref_emb": None, "face_landmarker": None, "device": "cpu"}
    mock_masker = mock_masker_cls.return_value
    mock_masker._create_frame_map.return_value = {1: "img1.jpg"}

    scenes = [Scene(shot_id=1, start_frame=1, end_frame=1)]

    with patch.object(pipeline, "_process_single_scene") as mock_process:
        result = pipeline.run(scenes)

    assert result == scenes
    assert mock_process.called
    assert mock_save_seeds.called


def test_pre_analysis_initialize_niqe_if_needed(mock_deps, analysis_params):
    from core.managers.analysis import PreAnalysisPipeline

    pipeline = PreAnalysisPipeline(
        mock_deps["config"],
        mock_deps["logger"],
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
        mock_deps["thumbnail_manager"],
        mock_deps["model_registry"],
    )

    analysis_params.pre_analysis_enabled = True
    analysis_params.video_path = "test.mp4"
    analysis_params.primary_seed_strategy = "Find Prominent Person"  # Matches strategy check

    with patch("pyiqa.create_metric") as mock_create:
        res = pipeline._initialize_niqe_if_needed("cpu", is_folder_mode=False)
        assert res == mock_create.return_value


@patch("core.managers.analysis.Image")
def test_pre_analysis_process_single_scene(mock_image, mock_deps, analysis_params, tmp_path):
    from core.managers.analysis import PreAnalysisPipeline

    analysis_params.output_folder = str(tmp_path)
    pipeline = PreAnalysisPipeline(
        mock_deps["config"],
        mock_deps["logger"],
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
        mock_deps["thumbnail_manager"],
        mock_deps["model_registry"],
    )

    scene = Scene(shot_id=1, start_frame=1, end_frame=1)
    scene.best_frame = 1  # Fix: Ensure best_frame is set so fname lookup works
    masker = MagicMock()
    masker.frame_map = {1: "frame_000001.webp"}
    masker.get_seed_for_frame.return_value = ([0, 0, 10, 10], {"score": 0.9})
    masker.get_mask_for_bbox.return_value = np.zeros((10, 10), dtype=np.uint8)

    mock_deps["thumbnail_manager"].get.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    previews_dir = tmp_path / "previews"
    previews_dir.mkdir()

    with patch("core.image_utils.render_mask_overlay", return_value=np.zeros((10, 10, 3), dtype=np.uint8)):
        pipeline._process_single_scene(scene, masker, previews_dir, is_folder_mode=False)

    assert scene.status == SceneStatus.INCLUDED
    assert "scene_00001.jpg" in scene.preview_path


def test_niqe_initialization_failures(mock_deps, analysis_params):
    pipeline = AnalysisPipeline(
        mock_deps["config"],
        mock_deps["logger"],
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
        mock_deps["thumbnail_manager"],
        mock_deps["model_registry"],
    )

    with patch("pyiqa.create_metric", side_effect=ImportError("mock")):
        pipeline._initialize_niqe_metric()
        assert pipeline.niqe_metric is None
        mock_deps["logger"].debug.assert_called()

    mock_deps["logger"].reset_mock()
    with patch("pyiqa.create_metric", side_effect=RuntimeError("mock")):
        pipeline._initialize_niqe_metric()
        assert pipeline.niqe_metric is None
        mock_deps["logger"].warning.assert_called()
