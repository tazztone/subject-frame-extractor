import threading
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch

import pytest

from core.events import ExtractionEvent, PreAnalysisEvent, PropagationEvent
from core.pipelines import (
    execute_analysis,
    execute_extraction,
    execute_pre_analysis,
    execute_propagation,
)


@pytest.fixture
def default_extraction_event():
    return {
        "source_path": "test.mp4",
        "method": "every_nth_frame",
        "interval": 1,
        "nth_frame": 1,
        "max_resolution": "1080",
        "thumb_megapixels": 0.5,
        "scene_detect": True,
        "output_folder": "/tmp/out",
    }


@pytest.fixture
def default_pre_analysis_event():
    return {
        "output_folder": "/tmp/out",
        "video_path": "test.mp4",
        "face_model_name": "buffalo_l",
        "tracker_model_name": "vit",
        "best_frame_strategy": "sharpness",
        "min_mask_area_pct": 0.1,
        "sharpness_base_scale": 1.0,
        "edge_strength_base_scale": 1.0,
        "pre_sample_nth": 1,
        "primary_seed_strategy": "Find Prominent Person",
    }


@pytest.fixture
def mock_runtime():
    return {
        "progress_queue": Queue(),
        "cancel_event": threading.Event(),
        "logger": MagicMock(),
        "config": MagicMock(downloads_dir="/tmp/downloads"),
        "thumbnail_manager": MagicMock(),
        "model_registry": MagicMock(),
    }


def test_execute_extraction_success(mock_runtime, tmp_path, default_extraction_event):
    mock_runtime["config"].downloads_dir = str(tmp_path / "downloads")
    Path(mock_runtime["config"].downloads_dir).mkdir()

    default_extraction_event["output_folder"] = str(tmp_path / "out")
    event = ExtractionEvent(**default_extraction_event)

    with patch("core.pipelines.ExtractionPipeline") as mock_pipeline_cls:
        mock_pipeline = mock_pipeline_cls.return_value
        mock_pipeline.run.return_value = {
            "done": True,
            "output_dir": str(tmp_path / "out"),
            "video_path": "test.mp4",
        }

        with patch("core.fingerprint.save_fingerprint"), patch("core.fingerprint.create_fingerprint"):
            gen = execute_extraction(
                event=event,
                progress_queue=mock_runtime["progress_queue"],
                cancel_event=mock_runtime["cancel_event"],
                logger=mock_runtime["logger"],
                config=mock_runtime["config"],
            )
            results = list(gen)

    assert len(results) == 1
    assert results[0]["done"] is True
    assert "Extraction Complete" in results[0]["unified_log"]
    assert results[0]["extracted_frames_dir_state"] == str(tmp_path / "out")


def test_execute_extraction_pipeline_failure(mock_runtime, default_extraction_event):
    event = ExtractionEvent(**default_extraction_event)

    with patch("core.pipelines.ExtractionPipeline") as mock_pipeline_cls:
        mock_pipeline = mock_pipeline_cls.return_value
        mock_pipeline.run.return_value = {"done": False, "log": "some error"}

        gen = execute_extraction(
            event=event,
            progress_queue=mock_runtime["progress_queue"],
            cancel_event=mock_runtime["cancel_event"],
            logger=mock_runtime["logger"],
            config=mock_runtime["config"],
        )
        results = list(gen)

    assert results[0]["done"] is False
    assert "Extraction failed: some error" in results[0]["unified_log"]


def test_execute_extraction_upload_copy(mock_runtime, tmp_path, default_extraction_event):
    downloads_dir = tmp_path / "downloads"
    downloads_dir.mkdir()
    mock_runtime["config"].downloads_dir = str(downloads_dir)

    # Create a dummy upload file
    upload_file = tmp_path / "upload.mp4"
    upload_file.write_text("dummy")

    default_extraction_event["upload_video"] = str(upload_file)
    event = ExtractionEvent(**default_extraction_event)

    with patch("core.pipelines.ExtractionPipeline") as mock_pipeline_cls:
        mock_pipeline = mock_pipeline_cls.return_value
        mock_pipeline.run.return_value = {"done": True, "output_dir": "out", "video_path": "v"}

        with patch("core.fingerprint.save_fingerprint"), patch("core.fingerprint.create_fingerprint"):
            gen = execute_extraction(
                event=event,
                progress_queue=mock_runtime["progress_queue"],
                cancel_event=mock_runtime["cancel_event"],
                logger=mock_runtime["logger"],
                config=mock_runtime["config"],
            )
            list(gen)

    assert (downloads_dir / "upload.mp4").exists()


def test_execute_pre_analysis_success(mock_runtime, tmp_path, default_pre_analysis_event):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    default_pre_analysis_event["output_folder"] = str(out_dir)
    event = PreAnalysisEvent(**default_pre_analysis_event)

    with (
        patch("core.pipelines.PreAnalysisPipeline") as mock_pipeline_cls,
        patch("core.pipelines._load_scenes", return_value=[]),
    ):
        mock_pipeline = mock_pipeline_cls.return_value
        mock_pipeline.run.return_value = []

        gen = execute_pre_analysis(
            event=event,
            progress_queue=mock_runtime["progress_queue"],
            cancel_event=mock_runtime["cancel_event"],
            logger=mock_runtime["logger"],
            config=mock_runtime["config"],
            thumbnail_manager=mock_runtime["thumbnail_manager"],
            cuda_available=True,
        )
        results = list(gen)

    assert results[0]["done"] is True
    assert "Pre-Analysis Complete" in results[0]["unified_log"]


def test_execute_propagation_no_scenes(mock_runtime, default_pre_analysis_event):
    pre_event = PreAnalysisEvent(**default_pre_analysis_event)
    event = PropagationEvent(output_folder="/tmp/out", video_path="test.mp4", scenes=[], analysis_params=pre_event)

    with patch("core.pipelines._load_analysis_scenes", return_value=[]):
        gen = execute_propagation(
            event=event,
            progress_queue=mock_runtime["progress_queue"],
            cancel_event=mock_runtime["cancel_event"],
            logger=mock_runtime["logger"],
            config=mock_runtime["config"],
            thumbnail_manager=mock_runtime["thumbnail_manager"],
            cuda_available=True,
        )
        results = list(gen)

    assert results[0]["done"] is True


def test_execute_analysis_success(mock_runtime, tmp_path, default_pre_analysis_event):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    pre_event = PreAnalysisEvent(**default_pre_analysis_event)
    event = PropagationEvent(
        output_folder=str(out_dir),
        video_path="test.mp4",
        scenes=[{"shot_id": 1, "start_frame": 0, "end_frame": 10}],
        analysis_params=pre_event,
    )

    with (
        patch("core.pipelines.AnalysisPipeline") as mock_pipeline_cls,
        patch("core.pipelines._load_analysis_scenes", return_value=[MagicMock()]),
    ):
        mock_pipeline = mock_pipeline_cls.return_value
        mock_pipeline.run_analysis_only.return_value = {"done": True, "output_dir": str(out_dir)}

        gen = execute_analysis(
            event=event,
            progress_queue=mock_runtime["progress_queue"],
            cancel_event=mock_runtime["cancel_event"],
            logger=mock_runtime["logger"],
            config=mock_runtime["config"],
            thumbnail_manager=mock_runtime["thumbnail_manager"],
            cuda_available=True,
        )
        results = list(gen)

    assert results[0]["done"] is True
    assert "metadata.db" in results[0]["metadata_path"]


def test_execute_pre_analysis_with_upload(mock_runtime, tmp_path, default_pre_analysis_event):
    downloads_dir = tmp_path / "downloads"
    downloads_dir.mkdir()
    mock_runtime["config"].downloads_dir = str(downloads_dir)

    upload_img = tmp_path / "face.jpg"
    upload_img.write_text("dummy")

    default_pre_analysis_event["face_ref_img_upload"] = str(upload_img)
    event = PreAnalysisEvent(**default_pre_analysis_event)

    with (
        patch("core.pipelines.PreAnalysisPipeline") as mock_pipeline_cls,
        patch("core.pipelines._load_scenes", return_value=[]),
    ):
        mock_pipeline = mock_pipeline_cls.return_value
        mock_pipeline.run.return_value = []

        gen = execute_pre_analysis(
            event=event,
            progress_queue=mock_runtime["progress_queue"],
            cancel_event=mock_runtime["cancel_event"],
            logger=mock_runtime["logger"],
            config=mock_runtime["config"],
            thumbnail_manager=mock_runtime["thumbnail_manager"],
            cuda_available=True,
        )
        list(gen)

    assert (downloads_dir / "face.jpg").exists()


def test_validate_session_dir_and_load(mock_runtime):
    from core.pipelines import execute_session_load, validate_session_dir

    with patch("core.pipelines._validate_session_dir", return_value=True):
        assert validate_session_dir("/tmp/session") is True

    with patch("core.pipelines._execute_session_load", return_value={"ok": True}):
        assert execute_session_load({}, mock_runtime["logger"]) == {"ok": True}


def test_execute_propagation_success(mock_runtime, tmp_path, default_pre_analysis_event):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    masks_dir = out_dir / "masks"
    masks_dir.mkdir()
    (masks_dir / "mask1.png").write_text("dummy")

    pre_event = PreAnalysisEvent(**default_pre_analysis_event)
    event = PropagationEvent(
        output_folder=str(out_dir), video_path="test.mp4", scenes=[{"shot_id": 1}], analysis_params=pre_event
    )

    with (
        patch("core.pipelines.AnalysisPipeline") as mock_pipeline_cls,
        patch("core.pipelines._load_analysis_scenes", return_value=[MagicMock()]),
        patch("core.pipelines.VideoManager.get_video_info", return_value={}),
        patch("core.pipelines.estimate_totals", return_value={"propagation": 10}),
    ):
        mock_pipeline = mock_pipeline_cls.return_value
        mock_pipeline.run_full_analysis.return_value = {"done": True, "output_dir": str(out_dir)}

        gen = execute_propagation(
            event=event,
            progress_queue=mock_runtime["progress_queue"],
            cancel_event=mock_runtime["cancel_event"],
            logger=mock_runtime["logger"],
            config=mock_runtime["config"],
            thumbnail_manager=mock_runtime["thumbnail_manager"],
            cuda_available=True,
        )
        results = list(gen)

    assert results[0]["done"] is True
    assert "1 masks generated" in results[0]["unified_log"]


def test_execute_propagation_failure(mock_runtime, default_pre_analysis_event):
    pre_event = PreAnalysisEvent(**default_pre_analysis_event)
    event = PropagationEvent(
        output_folder="/tmp/out", video_path="test.mp4", scenes=[{"shot_id": 1}], analysis_params=pre_event
    )

    with (
        patch("core.pipelines.AnalysisPipeline") as mock_pipeline_cls,
        patch("core.pipelines._load_analysis_scenes", return_value=[MagicMock()]),
        patch("core.pipelines.VideoManager.get_video_info", return_value={}),
        patch("core.pipelines.estimate_totals", return_value={}),
    ):
        mock_pipeline = mock_pipeline_cls.return_value
        mock_pipeline.run_full_analysis.return_value = {"done": False, "error": "failed"}

        gen = execute_propagation(
            event=event,
            progress_queue=mock_runtime["progress_queue"],
            cancel_event=mock_runtime["cancel_event"],
            logger=mock_runtime["logger"],
            config=mock_runtime["config"],
            thumbnail_manager=mock_runtime["thumbnail_manager"],
            cuda_available=True,
        )
        results = list(gen)

    assert results[0]["done"] is False
    assert "Propagation failed: failed" in results[0]["unified_log"]


def test_execute_analysis_failure(mock_runtime, default_pre_analysis_event):
    pre_event = PreAnalysisEvent(**default_pre_analysis_event)
    event = PropagationEvent(
        output_folder="/tmp/out", video_path="test.mp4", scenes=[{"shot_id": 1}], analysis_params=pre_event
    )

    with (
        patch("core.pipelines.AnalysisPipeline") as mock_pipeline_cls,
        patch("core.pipelines._load_analysis_scenes", return_value=[MagicMock()]),
    ):
        mock_pipeline = mock_pipeline_cls.return_value
        mock_pipeline.run_analysis_only.return_value = {"done": False, "error": "failed"}

        gen = execute_analysis(
            event=event,
            progress_queue=mock_runtime["progress_queue"],
            cancel_event=mock_runtime["cancel_event"],
            logger=mock_runtime["logger"],
            config=mock_runtime["config"],
            thumbnail_manager=mock_runtime["thumbnail_manager"],
            cuda_available=True,
        )
        results = list(gen)

    assert results[0]["done"] is False
    assert "Analysis failed: failed" in results[0]["unified_log"]


def test_execute_propagation_is_folder(mock_runtime, default_pre_analysis_event):
    default_pre_analysis_event["video_path"] = ""
    pre_event = PreAnalysisEvent(**default_pre_analysis_event)
    event = PropagationEvent(
        output_folder="/tmp/out", video_path="", scenes=[{"shot_id": 1}], analysis_params=pre_event
    )

    with (
        patch("core.pipelines.AnalysisPipeline") as mock_pipeline_cls,
        patch("core.pipelines._load_analysis_scenes", return_value=[MagicMock()]),
    ):
        mock_pipeline = mock_pipeline_cls.return_value
        mock_pipeline.run_full_analysis.return_value = {"done": True, "output_dir": "/tmp/out"}

        gen = execute_propagation(
            event=event,
            progress_queue=mock_runtime["progress_queue"],
            cancel_event=mock_runtime["cancel_event"],
            logger=mock_runtime["logger"],
            config=mock_runtime["config"],
            thumbnail_manager=mock_runtime["thumbnail_manager"],
            cuda_available=True,
        )
        list(gen)

    # Just verifying it doesn't crash and follows is_folder path


def test_fingerprint_failure_is_silent(mock_runtime, tmp_path, default_extraction_event):
    default_extraction_event["output_folder"] = str(tmp_path / "out")
    event = ExtractionEvent(**default_extraction_event)

    with patch("core.pipelines.ExtractionPipeline") as mock_pipeline_cls:
        mock_pipeline = mock_pipeline_cls.return_value
        mock_pipeline.run.return_value = {
            "done": True,
            "output_dir": str(tmp_path / "out"),
            "video_path": "test.mp4",
        }

        with patch("core.fingerprint.create_fingerprint", side_effect=RuntimeError("Fingerprint failed")):
            gen = execute_extraction(
                event=event,
                progress_queue=mock_runtime["progress_queue"],
                cancel_event=mock_runtime["cancel_event"],
                logger=mock_runtime["logger"],
                config=mock_runtime["config"],
            )
            results = list(gen)

    # Should still succeed even if fingerprinting fails
    assert results[0]["done"] is True
    mock_runtime["logger"].warning.assert_called()
