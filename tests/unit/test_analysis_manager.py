import json
import threading
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from core.managers.analysis import AnalysisPipeline, PreAnalysisPipeline, SceneStatus, _load_scenes
from core.models import AnalysisParameters, Scene


@pytest.fixture
def mock_deps():
    config = MagicMock()
    config.analysis_default_workers = 4
    config.analysis_default_batch_size = 1
    config.retry_max_attempts = 3
    config.retry_backoff_seconds = [1]
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
        tracker_model_name="sam2",
        best_frame_strategy="sharpness",
        min_mask_area_pct=0.1,
        sharpness_base_scale=1.0,
        edge_strength_base_scale=1.0,
        primary_seed_strategy="Automatic Detection",
    )


@patch("core.managers.analysis.Database")
def test_analysis_pipeline_initialization(mock_db, mock_deps, analysis_params):
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


def test_load_scenes_failure(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_scenes(tmp_path)


def test_load_scenes_success(tmp_path):
    scenes_path = tmp_path / "scenes.json"
    with scenes_path.open("w") as f:
        json.dump([[0, 10], [11, 20]], f)

    scenes = _load_scenes(tmp_path)
    assert len(scenes) == 2
    assert scenes[0].start_frame == 0
    assert scenes[1].end_frame == 20


@patch("core.managers.analysis.SubjectMasker")
@patch("core.managers.analysis.initialize_analysis_models")
@patch("core.managers.analysis.Database")
def test_run_full_analysis_success(mock_db, mock_init_models, mock_masker_cls, mock_deps, analysis_params, tmp_path):
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
    mock_init_models.return_value = {
        "face_analyzer": None,
        "ref_emb": None,
        "face_landmarker": None,
        "device": "cpu",
        "subject_detector": None,
    }

    result = pipeline.run_full_analysis(scenes)

    assert result["done"] is True
    assert mock_masker_cls.return_value.run_propagation.called


@patch("core.managers.analysis.run_operators")
@patch("core.managers.analysis.initialize_analysis_models")
@patch("core.managers.analysis.Database")
def test_run_analysis_only_success(mock_db, mock_init_models, mock_run_ops, mock_deps, analysis_params, tmp_path):
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

    mock_init_models.return_value = {
        "face_analyzer": None,
        "ref_emb": None,
        "face_landmarker": None,
        "device": "cpu",
        "subject_detector": None,
    }
    mock_run_ops.return_value = {"quality": MagicMock(success=True, metrics={"quality_score": 80})}

    # Need a frame_map in the pipeline
    pipeline.mask_metadata = {"frame_000001": {"mask_path": None}}

    # Mocking thumb_manager
    mock_deps["thumbnail_manager"].get.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

    with patch("core.managers.analysis.create_frame_map", return_value={1: "frame_000001.webp"}):
        scenes = [Scene(shot_id=1, start_frame=1, end_frame=1)]
        result = pipeline.run_analysis_only(scenes)

    assert result.get("done") is True
    assert mock_run_ops.called


@patch("core.managers.analysis.initialize_analysis_models")
@patch("core.managers.analysis.SubjectMasker")
def test_pre_analysis_run_cancellation(mock_masker_cls, mock_init_models, mock_deps, analysis_params, tmp_path):
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

    mock_init_models.return_value = {
        "face_analyzer": None,
        "ref_emb": None,
        "face_landmarker": None,
        "device": "cpu",
        "subject_detector": None,
    }
    mock_masker = mock_masker_cls.return_value
    mock_masker._create_frame_map.return_value = {1: "img1.jpg"}

    scenes = [Scene(shot_id=1, start_frame=1, end_frame=1)]
    mock_deps["cancel_event"].set()

    with patch("core.managers.analysis.save_scene_seeds"):
        result = pipeline.run(scenes)

    assert result == scenes
    # The break happens in the loop before processing
    assert not mock_masker.get_seed_for_frame.called


@patch("core.managers.analysis.initialize_analysis_models")
@patch("core.managers.analysis.Database")
def test_analysis_run_resume_logic(mock_db, mock_init_models, mock_deps, analysis_params, tmp_path):
    analysis_params.output_folder = str(tmp_path)
    analysis_params.resume = True

    # Create progress file
    progress_file = tmp_path / "progress.json"
    with progress_file.open("w") as f:
        json.dump({"completed_scenes": [1]}, f)

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
    mock_init_models.return_value = {
        "face_analyzer": None,
        "ref_emb": None,
        "face_landmarker": None,
        "device": "cpu",
        "subject_detector": None,
    }

    scenes = [Scene(shot_id=1, start_frame=0, end_frame=10), Scene(shot_id=2, start_frame=11, end_frame=20)]

    with patch("core.managers.analysis.SubjectMasker") as mock_masker_cls:
        pipeline.run_full_analysis(scenes)
        # Verify only scene 2 was passed to propagation
        args, _ = mock_masker_cls.return_value.run_propagation.call_args
        passed_scenes = args[1]
        assert len(passed_scenes) == 1
        assert passed_scenes[0].shot_id == 2


@patch("core.managers.analysis.cv2.imread")
@patch("core.managers.analysis.initialize_analysis_models")
@patch("core.managers.analysis.Database")
def test_process_reference_face_logic(mock_db, mock_init_models, mock_imread, mock_deps, analysis_params, tmp_path):
    analysis_params.output_folder = str(tmp_path)
    analysis_params.face_ref_img_path = str(tmp_path / "ref.jpg")
    (tmp_path / "ref.jpg").write_text("dummy")

    pipeline = AnalysisPipeline(
        mock_deps["config"],
        mock_deps["logger"],
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
        mock_deps["thumbnail_manager"],
        mock_deps["model_registry"],
    )

    # Mock face analyzer
    mock_face = MagicMock()
    mock_face.det_score = 0.9
    mock_face.normed_embedding = np.array([0.1, 0.2])
    pipeline.face_analyzer = MagicMock()
    pipeline.face_analyzer.get.return_value = [mock_face]
    mock_imread.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

    pipeline._process_reference_face()
    assert np.array_equal(pipeline.reference_embedding, mock_face.normed_embedding)


@patch("core.managers.analysis.run_operators")
@patch("core.managers.analysis.Database")
def test_process_single_frame_complex_meta(mock_db, mock_run_ops, mock_deps, analysis_params, tmp_path):
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
    pipeline.masks_dir = tmp_path / "masks"
    pipeline.masks_dir.mkdir()

    # Mock mask metadata with absolute path
    mask_path = tmp_path / "masks" / "frame_000001.png"
    mask_path.write_text("dummy")
    pipeline.mask_metadata = {"frame_000001": {"mask_path": str(mask_path), "shot_id": 1}}

    pipeline.scene_map = {1: Scene(shot_id=1, start_frame=0, end_frame=10)}
    pipeline.scene_map[1].seed_metrics = {"best_face_sim": 0.95}

    # Mock thumbnail and operator
    mock_deps["thumbnail_manager"].get.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_op_res = MagicMock(success=True, metrics={"sharpness": 0.5}, data={})
    mock_run_ops.return_value = {"sharpness": mock_op_res}

    with patch("core.managers.analysis.cv2.imread", return_value=np.zeros((10, 10), dtype=np.uint8)):
        pipeline._process_single_frame(Path("frame_000001.webp"), {"sharpness": True})

    assert pipeline.db.insert_metadata.called
    meta = pipeline.db.insert_metadata.call_args[0][0]
    assert meta["seed_face_sim"] == 0.95
    assert meta["mask_path"] == "frame_000001.png"


@patch("core.managers.analysis.Database")
def test_process_single_frame_face_analysis_failure(mock_db, mock_deps, analysis_params, tmp_path):
    """Test that face analysis failure is caught and logged."""
    from core.managers.analysis import AnalysisPipeline

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

    # Setup analyzer to fail
    mock_analyzer = MagicMock()
    mock_analyzer.get.side_effect = Exception("InsightFace Fail")
    pipeline.face_analyzer = mock_analyzer
    pipeline.params.compute_face_sim = True

    # Setup data
    img_path = Path("frame_000001.webp")
    mock_deps["thumbnail_manager"].get.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

    with patch("core.managers.analysis.cv2.imread", return_value=np.zeros((10, 10), dtype=np.uint8)):
        pipeline._process_single_frame(img_path, {})

    # Verify logger was called
    assert mock_deps["logger"].warning.called
    assert "Face analysis failed" in mock_deps["logger"].warning.call_args[0][0]


@patch("core.managers.analysis.Database")
def test_analysis_loop_batch_error(mock_db, mock_deps, analysis_params, tmp_path):
    pipeline = AnalysisPipeline(
        mock_deps["config"],
        mock_deps["logger"],
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
        mock_deps["thumbnail_manager"],
        mock_deps["model_registry"],
    )

    with (
        patch("core.managers.analysis.create_frame_map", return_value={1: "frame_000001.webp"}),
        patch("core.managers.analysis.ThreadPoolExecutor") as mock_executor_cls,
    ):
        mock_executor = mock_executor_cls.return_value.__enter__.return_value
        mock_future = MagicMock()
        mock_future.result.side_effect = Exception("Batch Fail")
        mock_executor.submit.return_value = mock_future

        # This should not raise but log the error
        pipeline._run_analysis_loop([Scene(shot_id=1, start_frame=1, end_frame=1)], {})

    assert mock_deps["logger"].error.called


@patch("core.managers.analysis.Database")
@patch("core.managers.analysis.create_frame_map")
def test_image_folder_analysis_trigger(mock_create_map, mock_db, mock_deps, analysis_params, tmp_path):
    analysis_params.video_path = ""
    analysis_params.output_folder = str(tmp_path)

    scenes_path = tmp_path / "scenes.json"
    with scenes_path.open("w") as f:
        json.dump([[0, 0]], f)

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

    with patch.object(pipeline, "run_analysis_only", return_value={"done": True}) as mock_run:
        res = pipeline.run_full_analysis([Scene(shot_id=0, start_frame=0, end_frame=0)])
        assert res["done"] is True
        assert mock_run.called


@patch("core.managers.analysis.Database")
def test_filter_completed_scenes(mock_db, mock_deps, analysis_params):
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


@patch("core.managers.analysis.Database")
def test_save_progress_bulk(mock_db, mock_deps, analysis_params, tmp_path):
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

    mock_init_models.return_value = {
        "face_analyzer": None,
        "ref_emb": None,
        "face_landmarker": None,
        "device": "cpu",
        "subject_detector": None,
    }
    mock_masker = mock_masker_cls.return_value
    mock_masker._create_frame_map.return_value = {1: "img1.jpg"}

    scenes = [Scene(shot_id=1, start_frame=1, end_frame=1)]

    with patch.object(pipeline, "_process_single_scene") as mock_process:
        result = pipeline.run(scenes)

    assert result == scenes
    assert mock_process.called
    assert mock_save_seeds.called


@patch("core.managers.analysis.Database")
def test_niqe_initialization_failures(mock_db, mock_deps, analysis_params):
    pipeline = AnalysisPipeline(
        mock_deps["config"],
        mock_deps["logger"],
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
        mock_deps["thumbnail_manager"],
        mock_deps["model_registry"],
    )

    # 1. ImportError
    with patch.dict("sys.modules", {"pyiqa": None}):
        # This will trigger ImportError inside _initialize_niqe_metric
        pipeline._initialize_niqe_metric()
        assert pipeline.niqe_metric is None
        mock_deps["logger"].debug.assert_called()

    # 2. Generic Exception
    with patch("pyiqa.create_metric", side_effect=Exception("NIQE failed")):
        pipeline._initialize_niqe_metric()
        assert pipeline.niqe_metric is None
        mock_deps["logger"].warning.assert_called()


def test_pre_analysis_process_single_scene_edge_cases(mock_deps, analysis_params, tmp_path):
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

    masker = MagicMock()
    masker.frame_map = {1: "f1.webp"}
    scene = Scene(shot_id=1, start_frame=1, end_frame=1)

    # 1. Thumbnail missing
    mock_deps["thumbnail_manager"].get.return_value = None
    pipeline._process_single_scene(scene, masker, tmp_path, is_folder_mode=False)
    assert scene.seed_result == {}

    # 2. Frame missing from map
    masker.frame_map = {}
    pipeline._process_single_scene(scene, masker, tmp_path, is_folder_mode=False)
    assert scene.seed_result == {}

    # 3. get_mask_for_bbox returns None
    masker.frame_map = {1: "f1.webp"}
    mock_deps["thumbnail_manager"].get.return_value = np.zeros((10, 10, 3))
    masker.get_seed_for_frame.return_value = ([0, 0, 5, 5], {"type": "test"})
    masker.get_mask_for_bbox.return_value = None
    analysis_params.enable_subject_mask = True
    pipeline._process_single_scene(scene, masker, tmp_path, is_folder_mode=False)
    assert "details" not in scene.seed_result or "mask_area_pct" not in scene.seed_result["details"]


@patch("core.managers.analysis.Database")
def test_process_single_frame_metadata_assembly(mock_db, mock_deps, analysis_params, tmp_path):
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
    # Use 'frame_000001' to match re.search(r"frame_(\d+)", path.name)
    pipeline.mask_metadata = {"frame_000001": {"mask_path": "m1.png", "shot_id": 1}}
    pipeline.scene_map = {1: Scene(shot_id=1, start_frame=1, end_frame=1, seed_metrics={"best_face_sim": 0.5})}

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_deps["thumbnail_manager"].get.return_value = img

    with (
        patch("core.managers.analysis.run_operators") as mock_run,
        patch("core.managers.analysis.cv2.imread", return_value=np.zeros((10, 10))),
        patch("core.managers.analysis.cv2.resize", return_value=np.zeros((10, 10))),
    ):
        mock_res = MagicMock()
        mock_res.success = True
        mock_res.metrics = {"face_sim": 0.8}
        mock_res.data = {"phash": "hash"}
        mock_run.return_value = {"op": mock_res}

        pipeline.params.compute_face_sim = True
        pipeline._process_single_frame(Path("frame_000001.webp"), {"op": True})

    assert pipeline.db.insert_metadata.called
    meta = pipeline.db.insert_metadata.call_args[0][0]
    assert meta["face_sim"] == 0.8
    assert meta["phash"] == "hash"
    assert meta["seed_face_sim"] == 0.5


@patch("core.managers.analysis.Database")
def test_save_progress_bulk_read_failure(mock_db, mock_deps, analysis_params, tmp_path):
    pipeline = AnalysisPipeline(
        mock_deps["config"],
        mock_deps["logger"],
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
        mock_deps["thumbnail_manager"],
        mock_deps["model_registry"],
    )
    progress_file = tmp_path / "corrupt_progress.json"
    with open(progress_file, "w") as f:
        f.write("{invalid json")

    pipeline._save_progress_bulk([1], progress_file)
    assert mock_deps["logger"].warning.called
    # Should still save if possible
    with open(progress_file, "r") as f:
        data = json.load(f)
        assert data["completed_scenes"] == [1]


@patch("core.managers.analysis.Database")
def test_process_reference_face_failures(mock_db, mock_deps, analysis_params, tmp_path):
    pipeline = AnalysisPipeline(
        mock_deps["config"],
        mock_deps["logger"],
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
        mock_deps["thumbnail_manager"],
        mock_deps["model_registry"],
    )
    analysis_params.face_ref_img_path = str(tmp_path / "nonexistent.jpg")
    pipeline.face_analyzer = MagicMock()
    with pytest.raises(FileNotFoundError):
        pipeline._process_reference_face()

    ref_img = tmp_path / "empty.jpg"
    ref_img.touch()
    analysis_params.face_ref_img_path = str(ref_img)
    with patch("cv2.imread", return_value=None):
        with pytest.raises(ValueError, match="Could not read ref image"):
            pipeline._process_reference_face()

    with patch("cv2.imread", return_value=np.zeros((10, 10, 3))):
        pipeline.face_analyzer = MagicMock()
        pipeline.face_analyzer.get.return_value = []
        with pytest.raises(ValueError, match="No face in ref image"):
            pipeline._process_reference_face()


@patch("core.managers.analysis.Database")
def test_process_single_frame_edge_cases(mock_db, mock_deps, analysis_params, tmp_path):
    # Mock SVD if needed, but safely using the existing global mock
    torch.linalg.svd.return_value = (None, torch.ones(1), None)
    pipeline = AnalysisPipeline(
        mock_deps["config"],
        mock_deps["logger"],
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
        mock_deps["thumbnail_manager"],
        mock_deps["model_registry"],
    )
    pipeline.mask_metadata = {"frame_000001": {"mask_path": "masks/m1.png"}}
    pipeline.masks_dir = tmp_path / "masks"
    pipeline.masks_dir.mkdir()

    # Match fails
    pipeline._process_single_frame(Path("not_a_frame.jpg"), {})

    # Img is None
    mock_deps["thumbnail_manager"].get.return_value = None
    pipeline._process_single_frame(Path("frame_000001.webp"), {})

    # Mask path not absolute
    mock_deps["thumbnail_manager"].get.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    (pipeline.masks_dir / "m1.png").touch()
    with patch("cv2.imread", return_value=np.zeros((100, 100), dtype=np.uint8)):
        pipeline._process_single_frame(Path("frame_000001.webp"), {})

    # Face analysis fails
    analysis_params.compute_face_sim = True
    pipeline.face_analyzer = MagicMock()
    pipeline.face_analyzer.get.side_effect = Exception("Face Fail")
    pipeline._process_single_frame(Path("frame_000001.webp"), {})
    assert mock_deps["logger"].warning.called


@patch("core.managers.analysis._load_scenes")
def test_image_folder_analysis_resume(mock_load, mock_deps, analysis_params, tmp_path):
    analysis_params.video_path = ""
    analysis_params.output_folder = str(tmp_path)
    analysis_params.resume = True

    pipeline = AnalysisPipeline(
        mock_deps["config"],
        mock_deps["logger"],
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
        mock_deps["thumbnail_manager"],
        mock_deps["model_registry"],
    )

    scenes = [
        Scene(shot_id=1, start_frame=1, end_frame=1, status=SceneStatus.INCLUDED),
        Scene(shot_id=2, start_frame=2, end_frame=2, status=SceneStatus.EXCLUDED),
    ]
    mock_load.return_value = scenes

    with patch.object(pipeline, "run_analysis_only") as mock_run:
        pipeline._run_image_folder_analysis()
        # Should only include shot_id 1
        processed_scenes = mock_run.call_args[0][0]
        assert len(processed_scenes) == 1
        assert processed_scenes[0].shot_id == 1


def test_pre_analysis_process_single_scene(mock_deps, analysis_params, tmp_path):
    from core.managers.analysis import PreAnalysisPipeline

    analysis_params.output_folder = str(tmp_path)
    analysis_params.enable_subject_mask = True

    pipeline = PreAnalysisPipeline(
        mock_deps["config"],
        mock_deps["logger"],
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
        mock_deps["thumbnail_manager"],
        mock_deps["model_registry"],
    )

    masker = MagicMock()
    masker.frame_map = {1: "frame_1.webp"}

    def side_effect(s, out):
        s.best_frame = 1

    masker._select_best_frame_in_scene.side_effect = side_effect

    masker.get_seed_for_frame.return_value = ([0, 0, 10, 10], {"conf": 0.9})
    masker.get_mask_for_bbox.return_value = np.ones((100, 100), dtype=np.uint8)

    mock_deps["thumbnail_manager"].get.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    scene = Scene(shot_id=1, start_frame=1, end_frame=1)
    previews_dir = tmp_path / "previews"
    previews_dir.mkdir()

    pipeline._process_single_scene(scene, masker, previews_dir, is_folder_mode=False)

    assert scene.status == SceneStatus.INCLUDED
    assert "mask_area_pct" in scene.seed_result["details"]
    assert Path(scene.preview_path).exists()


def test_initialize_niqe_success(mock_deps, analysis_params):
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
    from core.enums import SeedStrategy

    analysis_params.pre_analysis_enabled = True
    analysis_params.video_path = "vid.mp4"
    analysis_params.primary_seed_strategy = SeedStrategy.AUTOMATIC.value

    with patch("pyiqa.create_metric") as mock_create:
        pipeline._initialize_niqe_if_needed("cpu", is_folder_mode=False)
        assert mock_create.called


@patch("core.managers.analysis.Database")
def test_process_single_frame_full_meta(mock_db, mock_deps, analysis_params, tmp_path):
    pipeline = AnalysisPipeline(
        mock_deps["config"],
        mock_deps["logger"],
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
        mock_deps["thumbnail_manager"],
        mock_deps["model_registry"],
    )
    pipeline.mask_metadata = {"frame_000001": {"mask_path": "m1.png", "shot_id": 1}}
    pipeline.scene_map = {1: Scene(shot_id=1, start_frame=1, end_frame=1, seed_metrics={"best_face_sim": 0.85})}
    pipeline.db = MagicMock()

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_deps["thumbnail_manager"].get.return_value = img

    analysis_params.compute_face_sim = True
    pipeline.face_analyzer = MagicMock()
    mock_face = MagicMock()
    mock_face.bbox = np.array([0, 0, 10, 10])
    pipeline.face_analyzer.get.return_value = [mock_face]

    with patch("core.managers.analysis.run_operators") as mock_run:
        mock_res = MagicMock()
        mock_res.success = True
        mock_res.metrics = {"face_sim": 0.9, "quality": 0.8, "non_existent": 0.5}
        mock_res.data = {"phash": "hash"}
        mock_run.return_value = {"op1": mock_res}

        pipeline._process_single_frame(Path("frame_000001.webp"), {"quality": True})

    assert pipeline.db.insert_metadata.called
    args = pipeline.db.insert_metadata.call_args[0][0]
    assert args["face_sim"] == 0.9
    assert args["phash"] == "hash"
    assert args["seed_face_sim"] == 0.85
