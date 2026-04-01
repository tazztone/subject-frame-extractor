import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from core.events import ExportEvent
from core.export import (
    _crop_exported_frames,
    _export_metadata,
    _rename_exported_frames,
    dry_run_export,
    export_kept_frames,
)
from core.filtering import apply_all_filters_vectorized
from core.operators.crop import calculate_best_crop, crop_image_with_subject


class TestExportEdgeCases:
    @given(
        frame_w=st.integers(100, 4096),
        frame_h=st.integers(100, 4096),
        box_x=st.integers(0, 4000),
        box_y=st.integers(0, 4000),
        box_w=st.integers(1, 4000),
        box_h=st.integers(1, 4000),
        padding_factor=st.floats(1.0, 2.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_calculate_best_crop_bounds_safety(self, frame_w, frame_h, box_x, box_y, box_w, box_h, padding_factor):
        """Ensure that calculate_best_crop never returns a box exceeding frame dimensions or excluding subject."""
        # Ensure subject box is valid and within frame for this specific test logic
        box_x = min(box_x, frame_w - 1)
        box_y = min(box_y, frame_h - 1)
        box_w = max(1, min(box_w, frame_w - box_x))
        box_h = max(1, min(box_h, frame_h - box_y))

        subject_bbox = (box_x, box_y, box_w, box_h)
        aspect_ratios = [("1:1", 1.0), ("16:9", 16 / 9), ("9:16", 9 / 16)]

        crop = calculate_best_crop(frame_w, frame_h, subject_bbox, aspect_ratios, padding_factor)

        if crop:
            # Check bounds
            assert crop["x"] >= 0
            assert crop["y"] >= 0
            assert crop["x"] + crop["w"] <= frame_w
            assert crop["y"] + crop["h"] <= frame_h

            # Check subject containment (allowing for small float errors)
            assert crop["x"] <= subject_bbox[0] + 1
            assert crop["y"] <= subject_bbox[1] + 1
            assert crop["x"] + crop["w"] >= subject_bbox[0] + subject_bbox[2] - 1
            assert crop["y"] + crop["h"] >= subject_bbox[1] + subject_bbox[3] - 1

    def test_filtering_nan_quality_scores(self, mock_config):
        """Test filtering handles NaN or None quality scores without crashing."""
        frames = [
            {"filename": "f1.png", "metrics": {"quality_score": float("nan")}, "face_sim": np.nan},
            {"filename": "f2.png", "metrics": {"quality_score": 50.0}, "face_sim": 0.8},
        ]

        filters = {"quality_score_min": 40.0, "face_sim_min": 0.5, "face_sim_enabled": True}

        kept, rejected, counter, reasons = apply_all_filters_vectorized(frames, filters, mock_config)

        assert len(kept) == 1
        assert kept[0]["filename"] == "f2.png"
        assert len(rejected) == 1
        assert rejected[0]["filename"] == "f1.png"

    def test_filtering_extreme_thresholds(self, mock_config):
        """Test filtering with extreme thresholds."""
        frames = [{"filename": "f1.png", "metrics": {"quality_score": 50.0}, "mask_area_pct": 10.0}]

        # 1. Reject everything
        filters = {"quality_score_min": 100.0}
        kept, _, _, _ = apply_all_filters_vectorized(frames, filters, mock_config)
        assert len(kept) == 0

        # 2. Include everything
        filters = {"quality_score_min": 0.0, "mask_area_enabled": True, "mask_area_pct_min": 0.0}
        kept, _, _, _ = apply_all_filters_vectorized(frames, filters, mock_config)
        assert len(kept) == 1


class TestExportAdvanced:
    @patch("cv2.imread")
    @patch("cv2.imwrite")
    @patch("cv2.findContours")
    @patch("cv2.boundingRect")
    def test_crop_exported_frames_logic(
        self, mock_boundingRect, mock_findContours, mock_imwrite, mock_imread, tmp_path
    ):
        """Test the logic of cropping exported frames."""
        export_dir = tmp_path / "export"
        export_dir.mkdir()
        masks_root = tmp_path / "masks"
        masks_root.mkdir()

        # Setup fake files
        frame_name = "frame_000001.png"
        mask_name = "mask_000001.png"
        (export_dir / frame_name).touch()
        (masks_root / mask_name).touch()

        kept_frames = [{"filename": frame_name, "mask_path": mask_name}]

        # Mocks
        mock_imread.side_effect = [
            np.zeros((100, 100, 3), dtype=np.uint8),  # Frame
            np.zeros((100, 100), dtype=np.uint8),  # Mask
        ]
        mock_findContours.return_value = ([np.array([[[10, 10]], [[20, 20]]])], None)
        mock_boundingRect.return_value = (10, 10, 20, 20)  # x, y, w, h

        logger = MagicMock()
        cancel_event = MagicMock()
        cancel_event.is_set.return_value = False

        # Test 1:Basic Crop
        num_cropped = _crop_exported_frames(kept_frames, export_dir, "1:1", 0, masks_root, logger, cancel_event)
        assert num_cropped == 1
        mock_imwrite.assert_called()

        # Test 2: Invalid Aspect Ratio (must contain : but fail float conversion)
        with pytest.raises(ValueError):
            _crop_exported_frames(kept_frames, export_dir, "1:invalid", 0, masks_root, logger, cancel_event)

    @patch("cv2.imread")
    def test_crop_exported_frames_missing_files(self, mock_imread, tmp_path):
        """Test graceful handling of missing files."""
        export_dir = tmp_path / "export"
        export_dir.mkdir()
        masks_root = tmp_path / "masks"  # Exists but empty files
        masks_root.mkdir()

        kept_frames = [{"filename": "frame_000001.png", "mask_path": "mask_000001.png"}]
        logger = MagicMock()
        cancel_event = MagicMock()
        cancel_event.is_set.return_value = False

        num_cropped = _crop_exported_frames(kept_frames, export_dir, "1:1", 0, masks_root, logger, cancel_event)
        assert num_cropped == 0
        mock_imread.assert_not_called()

    @patch("cv2.imread")
    @patch("cv2.findContours")
    def test_crop_exported_frames_empty_mask(self, mock_findContours, mock_imread, tmp_path):
        """Test handling of empty masks."""
        export_dir = tmp_path / "export"
        export_dir.mkdir()
        masks_root = tmp_path / "masks"
        masks_root.mkdir()

        frame_name = "frame_000001.png"
        mask_name = "mask_000001.png"
        (export_dir / frame_name).touch()
        (masks_root / mask_name).touch()

        kept_frames = [{"filename": frame_name, "mask_path": mask_name}]

        mock_imread.side_effect = [np.zeros((100, 100, 3), dtype=np.uint8), np.zeros((100, 100), dtype=np.uint8)]
        # No contours found
        mock_findContours.return_value = ([], None)

        logger = MagicMock()
        cancel_event = MagicMock()
        cancel_event.is_set.return_value = False

        num_cropped = _crop_exported_frames(kept_frames, export_dir, "1:1", 0, masks_root, logger, cancel_event)
        assert num_cropped == 0


class TestExportExtended:
    """Extended tests for export.py to cover crop logic and edge cases."""

    @pytest.fixture
    def mock_deps(self, mock_config, mock_logger):
        return mock_config, mock_logger

    def test_crop_exported_frames_happy_path(self, mock_deps, tmp_path):
        config, logger = mock_deps
        export_dir = tmp_path / "export"
        export_dir.mkdir()
        (export_dir / "f1.png").touch()

        masks_root = tmp_path / "masks"
        masks_root.mkdir()
        (masks_root / "m1.png").touch()

        kept_frames = [{"filename": "f1.png", "mask_path": "m1.png"}]
        cancel_event = MagicMock()
        cancel_event.is_set.return_value = False

        with (
            patch("cv2.imread") as mock_read,
            patch("cv2.imwrite") as mock_write,
            patch("core.export.crop_image_with_subject") as mock_crop_logic,
        ):
            # Mock image and mask
            mock_read.side_effect = [
                np.zeros((100, 100, 3), dtype=np.uint8),  # frame
                np.zeros((100, 100), dtype=np.uint8),  # mask
            ]
            mock_crop_logic.return_value = (np.zeros((50, 50, 3)), "1x1")

            num = _crop_exported_frames(kept_frames, export_dir, "1:1", 10, masks_root, logger, cancel_event)

            assert num == 1
            mock_write.assert_called_once()

    def test_crop_exported_frames_invalid_ar(self, mock_deps, tmp_path):
        config, logger = mock_deps
        with pytest.raises(ValueError, match="Invalid aspect ratio format"):
            _crop_exported_frames([], tmp_path, "invalid", 10, tmp_path, logger, MagicMock())

    def test_crop_exported_frames_missing_files(self, mock_deps, tmp_path):
        config, logger = mock_deps
        kept_frames = [{"filename": "missing.png", "mask_path": "missing_mask.png"}]
        num = _crop_exported_frames(kept_frames, tmp_path, "1:1", 10, tmp_path, logger, MagicMock())
        assert num == 0

    def test_crop_exported_frames_read_failure(self, mock_deps, tmp_path):
        config, logger = mock_deps
        export_dir = tmp_path / "export"
        export_dir.mkdir()
        (export_dir / "f1.png").touch()
        (tmp_path / "m1.png").touch()

        kept_frames = [{"filename": "f1.png", "mask_path": "m1.png"}]
        with patch("core.export.cv2.imread", return_value=None):
            num = _crop_exported_frames(kept_frames, export_dir, "1:1", 10, tmp_path, logger, MagicMock())
            assert num == 0

    def test_crop_exported_frames_exception_handling(self, mock_deps, tmp_path):
        config, logger = mock_deps
        export_dir = tmp_path / "export"
        export_dir.mkdir()
        (export_dir / "f1.png").touch()
        (tmp_path / "m1.png").touch()

        kept_frames = [{"filename": "f1.png", "mask_path": "m1.png"}]
        cancel_event = MagicMock()
        cancel_event.is_set.return_value = False

        with patch("core.export.cv2.imread", side_effect=Exception("Read error")):
            num = _crop_exported_frames(kept_frames, export_dir, "1:1", 10, tmp_path, logger, cancel_event)
            assert num == 0
            logger.error.assert_called()

    @given(
        ar_w=st.floats(min_value=0.1, max_value=10.0),
        ar_h=st.floats(min_value=0.1, max_value=10.0),
        padding=st.integers(min_value=0, max_value=100),
    )
    def test_crop_logic_bounds_property(self, ar_w, ar_h, padding):
        """Property-based test for crop logic bounds (using internal operator)."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 255  # Subject in middle

        ars = [("test", ar_w / ar_h)]
        padding_factor = 1.0 + (padding / 100.0)

        cropped, _ = crop_image_with_subject(img, mask, ars, padding_factor)

        if cropped is not None:
            assert cropped.shape[0] <= 100
            assert cropped.shape[1] <= 100
            assert cropped.size > 0

    @patch("core.export.apply_all_filters_vectorized")
    def test_export_kept_frames_no_frame_map_error(self, mock_filter, mock_deps, tmp_path):
        config, logger = mock_deps
        mock_filter.return_value = ([{"filename": "f1.png"}], [], [], [])

        video_path = tmp_path / "v.mp4"
        video_path.touch()
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        event = ExportEvent(
            all_frames_data=[{"filename": "f1.png"}],
            video_path=str(video_path),
            output_dir=str(output_dir),
            filter_args={},
            enable_crop=False,
            crop_ars="1:1",
            crop_padding=10,
        )

        # Missing frame_map.json
        res = export_kept_frames(event, config, logger, None, None)
        assert "[ERROR] frame_map.json not found" in res

    @patch("core.export.apply_all_filters_vectorized")
    def test_export_kept_frames_folder_mode_happy_path(self, mock_filter, mock_deps, tmp_path):
        config, logger = mock_deps
        mock_filter.return_value = ([{"filename": "f1.png"}], [], [], [])

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        # Source file
        src_file = tmp_path / "original.jpg"
        src_file.write_text("data")

        # source_map.json
        source_map = {"f1.png": str(src_file)}
        with (output_dir / "source_map.json").open("w") as f:
            import json

            json.dump(source_map, f)

        event = ExportEvent(
            all_frames_data=[{"filename": "f1.png"}],
            video_path="",  # Folder mode
            output_dir=str(output_dir),
            filter_args={},
            enable_xmp_export=True,
            enable_crop=False,
            crop_ars="1:1",
            crop_padding=10,
        )

        with patch("core.xmp_writer.write_xmp_sidecar") as mock_xmp:
            res = export_kept_frames(event, config, logger, None, None)
            assert "Export Complete" in res
            assert mock_xmp.called


def test_rename_exported_frames_happy_path(mock_config, mock_logger, tmp_path):
    export_dir = tmp_path / "export"
    logger = mock_logger
    export_dir.mkdir()

    # Create fake extracted frames
    (export_dir / "frame_000001.webp").touch()
    (export_dir / "frame_000002.webp").touch()

    frames_to_extract = [10, 20]
    fn_to_orig_map = {"frame_000001.webp": 10, "frame_000002.webp": 20}

    _rename_exported_frames(export_dir, frames_to_extract, fn_to_orig_map, logger)

    # _rename_exported_frames logic:
    # fn_to_orig = { "frame_000001.webp": 10 }
    # orig_to_final = { 10: "frame_000001.webp" }
    # It tries to rename frame_000001.webp to its value in orig_to_final.
    # In this case it's already named that, so nothing happens.
    # Let's use different names.
    (export_dir / "frame_000001.webp").unlink()
    (export_dir / "frame_000002.webp").unlink()
    (export_dir / "frame_000001.webp").touch()
    (export_dir / "frame_000002.webp").touch()

    fn_to_orig_map = {"orig_10.png": 10, "orig_20.png": 20}
    # fn_to_orig maps desired name -> original frame index

    _rename_exported_frames(export_dir, frames_to_extract, fn_to_orig_map, logger)

    assert (export_dir / "orig_10.png").exists()
    assert (export_dir / "orig_20.png").exists()


def test_rename_exported_frames_collision(mock_config, mock_logger, tmp_path):
    export_dir = tmp_path / "export"
    logger = mock_logger
    export_dir.mkdir()

    (export_dir / "frame_000001.webp").touch()

    # Pre-existing file with the target name
    (export_dir / "orig_10.png").write_text("existing")

    frames_to_extract = [10]
    fn_to_orig_map = {"orig_10.png": 10}

    _rename_exported_frames(export_dir, frames_to_extract, fn_to_orig_map, logger)

    # Should rename to an alternate name like orig_10 (1).png
    assert (export_dir / "orig_10.png").read_text() == "existing"
    assert (export_dir / "orig_10 (1).png").exists()


@patch("pathlib.Path.rename")
def test_rename_exported_frames_missing_src(mock_rename, mock_config, mock_logger, tmp_path):
    export_dir = tmp_path / "export"
    logger = mock_logger
    export_dir.mkdir()

    (export_dir / "frame_000001.webp").touch()

    frames_to_extract = [10]
    fn_to_orig_map = {"target.png": 10}

    # Simulate rename failing with FileNotFoundError
    mock_rename.side_effect = FileNotFoundError

    _rename_exported_frames(export_dir, frames_to_extract, fn_to_orig_map, logger)

    logger.warning.assert_called()


def test_export_metadata_success(mock_config, mock_logger, tmp_path):
    export_dir = tmp_path / "export"
    logger = mock_logger
    export_dir.mkdir()

    kept_frames = [
        {"filename": "f1.png", "frame_number": 10, "score": 95.5, "extra": "data1"},
        {"filename": "f2.png", "timestamp": 1.5, "face_sim": 0.99, "extra": "data2"},
    ]

    _export_metadata(kept_frames, export_dir, logger)

    import csv

    # Check JSON
    json_file = export_dir / "metadata.json"
    assert json_file.exists()
    with json_file.open("r") as f:
        data = json.load(f)
        assert len(data) == 2
        assert data[0]["filename"] == "f1.png"

    # Check CSV
    csv_file = export_dir / "metadata.csv"
    assert csv_file.exists()
    with csv_file.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 2
        # Priority keys check
        assert reader.fieldnames[0] == "filename"
        assert rows[0]["filename"] == "f1.png"


def test_export_metadata_empty(mock_config, mock_logger, tmp_path):
    export_dir = tmp_path / "export"
    logger = mock_logger
    export_dir.mkdir()

    _export_metadata([], export_dir, logger)

    assert (export_dir / "metadata.json").exists()
    assert not (export_dir / "metadata.csv").exists()


def test_dry_run_export_no_data(mock_config, mock_logger):
    config, logger = mock_config, mock_logger

    event = ExportEvent(
        all_frames_data=[],
        video_path="v.mp4",
        output_dir="/tmp",
        filter_args={},
        enable_crop=False,
        crop_ars="",
        crop_padding=0,
    )
    res = dry_run_export(event, config, logger)
    assert res == "No metadata to export."


def test_dry_run_export_no_video(mock_config, mock_logger):
    config, logger = mock_config, mock_logger

    event = ExportEvent(
        all_frames_data=[{"filename": "f"}],
        video_path="",
        output_dir="/tmp",
        filter_args={},
        enable_crop=False,
        crop_ars="",
        crop_padding=0,
    )
    res = dry_run_export(event, config, logger)
    # Corrected expectation based on actual code
    assert "Mode: Folder" in res


@patch("core.export.apply_all_filters_vectorized")
def test_dry_run_export_success(mock_filter, mock_config, mock_logger, tmp_path):
    config, logger = mock_config, mock_logger

    video_path = tmp_path / "v.mp4"
    video_path.touch()

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with (out_dir / "frame_map.json").open("w") as f:
        json.dump([10, 20], f)

    mock_filter.return_value = ([{"filename": "frame_000001.webp"}], [], [], [])

    event = ExportEvent(
        all_frames_data=[{"filename": "frame_000001.webp"}],
        video_path=str(video_path),
        output_dir=str(out_dir),
        filter_args={},
        enable_crop=False,
        crop_ars="",
        crop_padding=0,
    )

    res = dry_run_export(event, config, logger)
    assert "Dry Run:" in res
    assert "ffmpeg" in res


@patch("core.export.apply_all_filters_vectorized")
def test_export_kept_frames_no_frames_kept(mock_filter, mock_config, mock_logger, tmp_path):
    config, logger = mock_config, mock_logger

    mock_filter.return_value = ([], [], [], [])

    video_path = tmp_path / "v.mp4"
    video_path.touch()

    event = ExportEvent(
        all_frames_data=[{"filename": "f1"}],
        video_path=str(video_path),
        output_dir=str(tmp_path),
        filter_args={},
        enable_crop=False,
        crop_ars="",
        crop_padding=0,
    )

    res = export_kept_frames(event, config, logger, None, None)
    assert "No frames kept" in res


@patch("core.export.apply_all_filters_vectorized")
@patch("core.export.perform_ffmpeg_export")
def test_export_kept_frames_ffmpeg_failure(mock_ffmpeg, mock_filter, mock_config, mock_logger, tmp_path):
    config, logger = mock_config, mock_logger

    video_path = tmp_path / "v.mp4"
    video_path.touch()

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    with (out_dir / "frame_map.json").open("w") as f:
        json.dump([10], f)

    mock_filter.return_value = ([{"filename": "frame_000001.webp"}], [], [], [])
    mock_ffmpeg.return_value = (False, "FFmpeg error")

    event = ExportEvent(
        all_frames_data=[{"filename": "frame_000001.webp"}],
        video_path=str(video_path),
        output_dir=str(out_dir),
        filter_args={},
        enable_crop=False,
        crop_ars="",
        crop_padding=0,
    )

    res = export_kept_frames(event, config, logger, None, None)
    assert "FFmpeg failed" in res


@patch("core.export.apply_all_filters_vectorized")
def test_dry_run_export_no_frames_kept(mock_filter, mock_config, mock_logger, tmp_path):
    config, logger = mock_config, mock_logger

    video_path = tmp_path / "v.mp4"
    video_path.touch()

    mock_filter.return_value = ([], [], [], [])

    event = ExportEvent(
        all_frames_data=[{"filename": "f1"}],
        video_path=str(video_path),
        output_dir=str(tmp_path),
        filter_args={},
        enable_crop=False,
        crop_ars="",
        crop_padding=0,
    )

    res = dry_run_export(event, config, logger)
    assert "No frames kept" in res


def test_write_xmp_sidecar_rename_failure(tmp_path):
    from core.xmp_writer import write_xmp_sidecar

    dest_path = tmp_path / "test.jpg"
    dest_path.touch()

    with patch("core.xmp_writer.logger") as mock_xmp_logger:
        with patch("os.replace", side_effect=OSError("Atomic rename failed")):
            # This shouldn't crash but log error
            write_xmp_sidecar(dest_path, 5, "Green")
            mock_xmp_logger.error.assert_called()


@patch("core.export.cv2.imread", return_value=None)
def test_crop_exported_frames_folder_mode_read_failure(mock_read, mock_logger, tmp_path):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    frame_path = export_dir / "f1.png"
    frame_path.touch()
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    mask_file = masks_dir / "m1.png"
    mask_file.touch()

    kept_frames = [{"filename": "f1.png", "mask_path": "m1.png"}]
    cancel_event = MagicMock()
    cancel_event.is_set.return_value = False

    num = _crop_exported_frames(kept_frames, export_dir, "1:1", 10, masks_dir, mock_logger, cancel_event)
    assert num == 0
    # Search all calls for 'error'
    assert any(c[0] == "error" for c in mock_logger.method_calls), f"Calls: {mock_logger.method_calls}"
