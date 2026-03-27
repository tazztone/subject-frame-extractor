import json
from unittest.mock import MagicMock, patch

from core.events import ExportEvent
from core.export import _export_metadata, export_kept_frames


def test_export_metadata_logic(tmp_path, mock_logger):
    """Test the internal _export_metadata function."""
    export_dir = tmp_path / "export_meta"
    export_dir.mkdir()

    kept_frames = [
        {"filename": "f1.png", "frame_number": 10, "score": 95.0, "metrics": {"brightness": 128}},
        {"filename": "f2.png", "frame_number": 20, "score": 85.0, "metrics": {"brightness": 100}},
    ]

    _export_metadata(kept_frames, export_dir, mock_logger)

    # Check JSON
    json_path = export_dir / "metadata.json"
    assert json_path.exists()
    with json_path.open() as f:
        data = json.load(f)
        assert len(data) == 2
        assert data[0]["filename"] == "f1.png"

    # Check CSV
    csv_path = export_dir / "metadata.csv"
    assert csv_path.exists()
    content = csv_path.read_text()
    assert "filename,frame_number" in content
    assert "f1.png,10" in content


def test_export_kept_frames_no_data(mock_config, mock_logger):
    """Test export with no data."""
    event = ExportEvent(
        all_frames_data=[],
        output_dir="/tmp",
        video_path="dummy.mp4",
        filter_args={},
        enable_crop=False,
        crop_ars="1:1",
        crop_padding=10,
    )
    result = export_kept_frames(event, mock_config, mock_logger, MagicMock(), MagicMock())
    assert "No metadata to export" in result


def test_export_kept_frames_folder_mode(tmp_path, mock_config, mock_logger):
    """Test export in folder mode (copying files)."""
    out_root = tmp_path / "output"
    out_root.mkdir()

    # Mock source files
    src_dir = tmp_path / "source"
    src_dir.mkdir()
    f1 = src_dir / "orig1.jpg"
    f1.write_text("fake image")

    # Mock source_map.json
    source_map = {"frame_000001.webp": str(f1)}
    with (out_root / "source_map.json").open("w") as f:
        json.dump(source_map, f)

    event = ExportEvent(
        all_frames_data=[{"filename": "frame_000001.webp", "score": 90.0}],
        output_dir=str(out_root),
        video_path="",  # Folder mode
        filter_args={"enable_dedup": False},
        enable_xmp_export=True,
        enable_crop=False,
        crop_ars="1:1",
        crop_padding=10,
    )
    with patch("core.export.write_xmp_sidecar") as mock_xmp:
        result = export_kept_frames(event, mock_config, mock_logger, MagicMock(), MagicMock())
        assert "✅ Export Complete" in result
        assert mock_xmp.called

        # Verify file was copied to the export dir
        # The export dir name includes a timestamp, so we look for it
        export_dirs = list(tmp_path.glob("output_exported_*"))
        assert len(export_dirs) == 1
        assert (export_dirs[0] / "orig1.jpg").exists()
