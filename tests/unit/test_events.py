import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.enums import SceneStatus
from core.events import (
    ExportEvent,
    ExtractionEvent,
    FilterEvent,
    PreAnalysisEvent,
    PropagationEvent,
    SessionLoadEvent,
    UIEvent,
    validate_writable_directory,
)


def test_validate_writable_directory():
    """Test the validate_writable_directory helper."""
    # Empty path
    assert validate_writable_directory("", "Test Field") == ""

    # Existing writable directory
    with patch("core.events.Path.exists", return_value=True), \
         patch("os.access", return_value=True):
        assert validate_writable_directory("/valid/path", "Test Field") == "/valid/path"

    # Existing non-writable directory
    with patch("core.events.Path.exists", return_value=True), \
         patch("os.access", return_value=False):
        with pytest.raises(ValueError, match="Test Field '/invalid/path' is not writable"):
            validate_writable_directory("/invalid/path", "Test Field")

    # Non-existing directory with writable parent
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = False
    mock_path.parent.exists.return_value = True
    with patch("core.events.Path", return_value=mock_path), \
         patch("os.access", return_value=True):
        assert validate_writable_directory("/new/dir", "Test Field") == "/new/dir"

    # Non-existing directory with non-writable parent
    mock_path = MagicMock(spec=Path)
    mock_path.exists.return_value = False
    mock_path.parent.exists.return_value = True
    with patch("core.events.Path", return_value=mock_path), \
         patch("os.access", return_value=False):
        with pytest.raises(ValueError, match="Test Field '/new/dir' is not writable"):
            validate_writable_directory("/new/dir", "Test Field")


def test_extraction_event_validation():
    """Test ExtractionEvent validation logic."""
    # Valid event
    event = ExtractionEvent(
        source_path="video.mp4",
        method="all",
        interval=5.0,
        nth_frame=5,
        max_resolution="1080p",
        thumb_megapixels=0.5,
        scene_detect=True,
        output_folder="out",
    )
    assert event.source_path == "video.mp4"

    # Missing both source_path and upload_video
    with pytest.raises(ValueError, match="Please provide a Source Path or Upload a Video"):
        ExtractionEvent(
            source_path="",
            upload_video=None,
            method="all",
            interval=5.0,
            nth_frame=5,
            max_resolution="1080p",
            thumb_megapixels=0.5,
            scene_detect=True,
            output_folder="out",
        )

    # Test output_folder validation
    with patch("core.events.validate_writable_directory", return_value="out_validated"):
        event = ExtractionEvent(
            source_path="video.mp4",
            method="all",
            max_resolution="1080p",
            scene_detect=True,
            output_folder="out",
        )
        assert event.output_folder == "out_validated"

        # Test output_folder None (Line 55)
        event_none = ExtractionEvent(
            source_path="video.mp4",
            method="all",
            max_resolution="1080p",
            scene_detect=True,
            output_folder=None,
        )
        assert event_none.output_folder is None


def test_pre_analysis_event_validation():
    """Test PreAnalysisEvent validation logic."""
    common_args = {
        "output_folder": "out",
        "video_path": "video.mp4",
        "face_model_name": "buffalo_l",
        "tracker_model_name": "sam3",
        "best_frame_strategy": "Largest Person",
        "min_mask_area_pct": 1.0,
        "sharpness_base_scale": 2500.0,
        "edge_strength_base_scale": 100.0,
        "primary_seed_strategy": "Find Prominent Person",
        "pre_sample_nth": 1,
    }

    # Valid event
    event = PreAnalysisEvent(**common_args)
    assert event.output_folder == "out"

    # Test output_folder validation (validate_out)
    with patch("core.events.validate_writable_directory", return_value="out_validated"):
        event = PreAnalysisEvent(**common_args)
        assert event.output_folder == "out_validated"

    # Test face_ref_img_path validation (invalid path)
    with patch("core.events.Path.is_file", return_value=False):
        event_with_face = PreAnalysisEvent(**common_args, compute_face_sim=True, face_ref_img_path="nonexistent.jpg")
        assert event_with_face.compute_face_sim is True
        assert event_with_face.face_ref_img_path == ""

    # Test face_ref_img_path validation (valid path)
    with patch("core.events.Path.is_file", return_value=True):
        event_with_face = PreAnalysisEvent(**common_args, compute_face_sim=True, face_ref_img_path="existent.jpg")
        assert event_with_face.face_ref_img_path == "existent.jpg"

    # Test face_ref_img_path empty (Line 125)
    event_empty_face = PreAnalysisEvent(**common_args, face_ref_img_path="")
    assert event_empty_face.face_ref_img_path == ""

    # Test face_ref_img_path same as video_path
    event_same = PreAnalysisEvent(**common_args, face_ref_img_path="video.mp4")
    assert event_same.face_ref_img_path == ""

    # Test face_ref_img_path invalid extension
    with patch("core.events.Path.is_file", return_value=True):
        event_ext = PreAnalysisEvent(**common_args, face_ref_img_path="file.txt")
        assert event_ext.face_ref_img_path == ""

    # Test emoji stripping from strategy
    emoji_args = common_args.copy()
    emoji_args["primary_seed_strategy"] = "👤 Person"
    event_emoji = PreAnalysisEvent(**emoji_args)
    assert event_emoji.primary_seed_strategy == "Person"

    no_str_args = common_args.copy()
    no_str_args["primary_seed_strategy"] = 123
    event_no_str = PreAnalysisEvent(**no_str_args)
    assert event_no_str.primary_seed_strategy == "123"


def test_propagation_event_validation():
    """Test PropagationEvent validation."""
    analysis_params = PreAnalysisEvent(
        output_folder="out",
        video_path="video.mp4",
        face_model_name="buffalo_l",
        tracker_model_name="sam3",
        best_frame_strategy="sharpness",
    )

    # Valid event
    with patch("core.events.validate_writable_directory", return_value="out"):
        event = PropagationEvent(
            output_folder="out",
            video_path="video.mp4",
            scenes=[
                {"id": 1, "status": "included"},
                {"id": 2, "status": SceneStatus.EXCLUDED},  # Not a string
                {"id": 3}  # No status
            ],
            analysis_params=analysis_params
        )
        assert event.output_folder == "out"
        assert event.scenes[0]["status"] == SceneStatus.INCLUDED
        assert event.scenes[1]["status"] == SceneStatus.EXCLUDED
        assert "status" not in event.scenes[2]

    # Test invalid status fallback
    event_invalid = PropagationEvent(
        output_folder="out",
        video_path="video.mp4",
        scenes=[{"id": 1, "status": "unknown"}],
        analysis_params=analysis_params
    )
    assert event_invalid.scenes[0]["status"] == "unknown"


def test_filter_event_validation():
    """Test FilterEvent validation."""
    with patch("core.events.validate_writable_directory", return_value="out"):
        event = FilterEvent(
            all_frames_data=[],
            per_metric_values={},
            output_dir="out",
            gallery_view="grid",
            show_overlay=True,
            overlay_alpha=0.5,
            require_face_match=False,
            dedup_thresh=50,
            slider_values={},
            dedup_method="phash"
        )
        assert event.output_dir == "out"


def test_export_event_validation():
    """Test ExportEvent validation."""
    with patch("core.events.validate_writable_directory", return_value="out"):
        event = ExportEvent(
            all_frames_data=[],
            output_dir="out",
            video_path="video.mp4",
            enable_crop=False,
            crop_ars="1:1",
            crop_padding=0,
            filter_args={}
        )
        assert event.output_dir == "out"


def test_session_load_event_validation():
    """Test SessionLoadEvent validation."""
    # Empty path
    event = SessionLoadEvent(session_path="")
    assert event.session_path == ""

    # Valid path
    with patch("core.events.Path.exists", return_value=True), \
         patch("core.events.Path.is_dir", return_value=True):
        event = SessionLoadEvent(session_path="valid/session")
        assert event.session_path == "valid/session"

    # Non-existent path
    with patch("core.events.Path.exists", return_value=False):
        with pytest.raises(ValueError, match="Session path 'invalid/path' does not exist"):
            SessionLoadEvent(session_path="invalid/path")

    # Path is not a directory
    with patch("core.events.Path.exists", return_value=True), \
         patch("core.events.Path.is_dir", return_value=False):
        with pytest.raises(ValueError, match="Session path 'file/path' is not a directory"):
            SessionLoadEvent(session_path="file/path")


def test_ui_event_extra_ignore():
    """Test that UIEvent ignores extra fields as configured."""
    event = UIEvent(extra_field="ignore me")
    # Should not raise error and should not have extra_field
    assert not hasattr(event, "extra_field")
