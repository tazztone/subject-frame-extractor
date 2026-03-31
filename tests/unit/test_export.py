from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from core.events import ExportEvent
from core.export import _crop_exported_frames, export_kept_frames


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
        from core.operators.crop import crop_image_with_subject

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

        with patch("core.export.write_xmp_sidecar") as mock_xmp:
            res = export_kept_frames(event, config, logger, None, None)
            assert "Export Complete" in res
            assert mock_xmp.called

def test_rename_exported_frames_happy_path(mock_config, mock_logger, tmp_path):
    from core.export import _rename_exported_frames
    export_dir = tmp_path / "export"
    logger = mock_logger
    export_dir.mkdir()

    # Create fake extracted frames
    (export_dir / "frame_000001.png").touch()
    (export_dir / "frame_000002.png").touch()

    frames_to_extract = [10, 20]
    fn_to_orig_map = {"orig_10.png": 10, "orig_20.png": 20}

    _rename_exported_frames(export_dir, frames_to_extract, fn_to_orig_map, logger)

    assert (export_dir / "orig_10.png").exists()
    assert (export_dir / "orig_20.png").exists()
    assert not (export_dir / "frame_000001.png").exists()
    assert not (export_dir / "frame_000002.png").exists()

def test_rename_exported_frames_collision(mock_config, mock_logger, tmp_path):
    from core.export import _rename_exported_frames
    export_dir = tmp_path / "export"
    logger = mock_logger
    export_dir.mkdir()

    (export_dir / "frame_000001.png").touch()

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
    from core.export import _rename_exported_frames
    export_dir = tmp_path / "export"
    logger = mock_logger
    export_dir.mkdir()

    (export_dir / "frame_000001.png").touch()

    frames_to_extract = [10]
    fn_to_orig_map = {"target.png": 10}

    # Simulate rename failing with FileNotFoundError
    mock_rename.side_effect = FileNotFoundError

    _rename_exported_frames(export_dir, frames_to_extract, fn_to_orig_map, logger)

    logger.warning.assert_called()

def test_export_metadata_success(mock_config, mock_logger, tmp_path):
    from core.export import _export_metadata
    export_dir = tmp_path / "export"
    logger = mock_logger
    export_dir.mkdir()

    kept_frames = [
        {"filename": "f1.png", "frame_number": 10, "score": 95.5, "extra": "data1"},
        {"filename": "f2.png", "timestamp": 1.5, "face_sim": 0.99, "extra": "data2"}
    ]

    _export_metadata(kept_frames, export_dir, logger)

    import csv
    import json

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
    from core.export import _export_metadata
    export_dir = tmp_path / "export"
    logger = mock_logger
    export_dir.mkdir()

    _export_metadata([], export_dir, logger)

    assert (export_dir / "metadata.json").exists()
    assert not (export_dir / "metadata.csv").exists()

def test_dry_run_export_no_data(mock_config, mock_logger):
    from core.events import ExportEvent
    from core.export import dry_run_export
    config, logger = mock_config, mock_logger

    event = ExportEvent(all_frames_data=[], video_path="v.mp4", output_dir="/tmp", filter_args={}, enable_crop=False, crop_ars="", crop_padding=0)
    res = dry_run_export(event, config, logger)
    assert res == "No metadata to export."

def test_dry_run_export_no_video(mock_config, mock_logger):
    from core.events import ExportEvent
    from core.export import dry_run_export
    config, logger = mock_config, mock_logger

    event = ExportEvent(all_frames_data=[{"filename": "f"}], video_path="", output_dir="/tmp", filter_args={}, enable_crop=False, crop_ars="", crop_padding=0)
    res = dry_run_export(event, config, logger)
    assert "[ERROR] Original video path is required" in res

@patch("core.export.apply_all_filters_vectorized")
def test_dry_run_export_success(mock_filter, mock_config, mock_logger, tmp_path):
    from core.events import ExportEvent
    from core.export import dry_run_export
    config, logger = mock_config, mock_logger

    video_path = tmp_path / "v.mp4"
    video_path.touch()

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    import json
    with (out_dir / "frame_map.json").open("w") as f:
        json.dump([10, 20], f)

    mock_filter.return_value = ([{"filename": "frame_000001.webp"}], [], [], [])

    event = ExportEvent(
        all_frames_data=[{"filename": "frame_000001.webp"}],
        video_path=str(video_path),
        output_dir=str(out_dir),
        filter_args={},
        enable_crop=False, crop_ars="", crop_padding=0
    )

    res = dry_run_export(event, config, logger)
    assert "Dry Run:" in res
    assert "ffmpeg" in res

@patch("core.export.apply_all_filters_vectorized")
def test_export_kept_frames_no_frames_kept(mock_filter, mock_config, mock_logger, tmp_path):
    from core.events import ExportEvent
    from core.export import export_kept_frames
    config, logger = mock_config, mock_logger

    mock_filter.return_value = ([], [], [], [])

    video_path = tmp_path / "v.mp4"
    video_path.touch()

    event = ExportEvent(
        all_frames_data=[{"filename": "f1"}],
        video_path=str(video_path),
        output_dir=str(tmp_path),
        filter_args={},
        enable_crop=False, crop_ars="", crop_padding=0
    )

    res = export_kept_frames(event, config, logger, None, None)
    assert "No frames kept" in res

@patch("core.export.apply_all_filters_vectorized")
@patch("core.export.perform_ffmpeg_export")
def test_export_kept_frames_ffmpeg_failure(mock_ffmpeg, mock_filter, mock_config, mock_logger, tmp_path):
    from core.events import ExportEvent
    from core.export import export_kept_frames
    config, logger = mock_config, mock_logger

    video_path = tmp_path / "v.mp4"
    video_path.touch()

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    import json
    with (out_dir / "frame_map.json").open("w") as f:
        json.dump([10], f)

    mock_filter.return_value = ([{"filename": "frame_000001.webp"}], [], [], [])
    mock_ffmpeg.return_value = (False, "FFmpeg error")

    event = ExportEvent(
        all_frames_data=[{"filename": "frame_000001.webp"}],
        video_path=str(video_path),
        output_dir=str(out_dir),
        filter_args={},
        enable_crop=False, crop_ars="", crop_padding=0
    )

    res = export_kept_frames(event, config, logger, None, None)
    assert "FFmpeg failed" in res
