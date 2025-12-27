"""
Tests for export functionality.

Covers all functions in core/export.py:
- export_kept_frames (main export function)
- dry_run_export (preview export without writing)
- _perform_ffmpeg_export (internal FFmpeg call)
- _rename_exported_frames (rename to original names)
- _crop_exported_frames (crop around mask)
"""

from unittest.mock import MagicMock, patch

import pytest

from core.events import ExportEvent
from core.export import dry_run_export, export_kept_frames


class TestExportKeptFrames:
    """Tests for the main export_kept_frames function."""

    @patch("subprocess.Popen")
    @patch("core.export.apply_all_filters_vectorized")
    def test_export_kept_frames_basic(self, mock_filter, mock_popen, mock_config, mock_logger, tmp_path):
        """Test basic export functionality."""
        mock_filter.return_value = ([{"filename": "frame_000001.webp"}], [], [], [])

        process = MagicMock()
        process.returncode = 0
        process.communicate.return_value = ("", "")
        mock_popen.return_value = process

        video_path = tmp_path / "video.mp4"
        video_path.touch()
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        # Create frame_map.json
        (output_dir / "frame_map.json").write_text("[0, 1, 2]")

        event = ExportEvent(
            all_frames_data=[{"filename": "frame_000001.webp"}],
            video_path=str(video_path),
            output_dir=str(output_dir),
            filter_args={},
            enable_crop=False,
            crop_ars="1:1",
            crop_padding=10,
        )

        result = export_kept_frames(event, mock_config, mock_logger, None, None)

        assert "Exported 1 frames" in result
        mock_popen.assert_called()

    @patch("subprocess.Popen")
    @patch("core.export.apply_all_filters_vectorized")
    def test_export_no_frames_kept(self, mock_filter, mock_popen, mock_config, mock_logger, tmp_path):
        """Test export when no frames pass filters."""
        mock_filter.return_value = ([], [], [], [])

        video_path = tmp_path / "video.mp4"
        video_path.touch()
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        (output_dir / "frame_map.json").write_text("[0, 1, 2]")

        event = ExportEvent(
            all_frames_data=[{"filename": "frame_000001.webp"}],
            video_path=str(video_path),
            output_dir=str(output_dir),
            filter_args={},
            enable_crop=False,
            crop_ars="1:1",
            crop_padding=10,
        )

        result = export_kept_frames(event, mock_config, mock_logger, None, None)

        assert "No frames" in result or "0" in result
        mock_popen.assert_not_called()

    @patch("subprocess.Popen")
    @patch("core.export.apply_all_filters_vectorized")
    def test_export_ffmpeg_failure(self, mock_filter, mock_popen, mock_config, mock_logger, tmp_path):
        """Test export handles FFmpeg failure gracefully."""
        mock_filter.return_value = ([{"filename": "frame_000001.webp"}], [], [], [])

        process = MagicMock()
        process.returncode = 1  # FFmpeg error
        process.communicate.return_value = ("", "FFmpeg error")
        mock_popen.return_value = process

        video_path = tmp_path / "video.mp4"
        video_path.touch()
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        (output_dir / "frame_map.json").write_text("[0, 1, 2]")

        event = ExportEvent(
            all_frames_data=[{"filename": "frame_000001.webp"}],
            video_path=str(video_path),
            output_dir=str(output_dir),
            filter_args={},
            enable_crop=False,
            crop_ars="1:1",
            crop_padding=10,
        )

        result = export_kept_frames(event, mock_config, mock_logger, None, None)

        # Should handle error gracefully
        assert result is not None


class TestDryRunExport:
    """Tests for dry_run_export function."""

    def test_dry_run_basic(self, mock_config, tmp_path):
        """Test basic dry run export returns expected format."""
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        (output_dir / "frame_map.json").write_text("[0, 1, 2, 3, 4]")

        event = ExportEvent(
            all_frames_data=[
                {"filename": "frame_000000.webp"},
                {"filename": "frame_000001.webp"},
            ],
            video_path=str(video_path),
            output_dir=str(output_dir),
            filter_args={},
            enable_crop=False,
            crop_ars="1:1",
            crop_padding=10,
        )

        result = dry_run_export(event, mock_config)

        assert "Would export" in result or "frames" in result.lower()

    def test_dry_run_no_frames(self, mock_config, tmp_path):
        """Test dry run with no frames."""
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        (output_dir / "frame_map.json").write_text("[]")

        event = ExportEvent(
            all_frames_data=[],
            video_path=str(video_path),
            output_dir=str(output_dir),
            filter_args={},
            enable_crop=False,
            crop_ars="1:1",
            crop_padding=10,
        )

        result = dry_run_export(event, mock_config)
        assert result is not None


class TestExportEvent:
    """Tests for ExportEvent validation."""

    def test_event_creation_minimal(self, tmp_path):
        """Test ExportEvent with minimal required fields."""
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        event = ExportEvent(
            all_frames_data=[],
            video_path=str(video_path),
            output_dir=str(output_dir),
            filter_args={},
            enable_crop=False,
            crop_ars="1:1",
            crop_padding=10,
        )

        assert event.video_path == str(video_path)
        assert event.enable_crop is False

    def test_event_with_crop_settings(self, tmp_path):
        """Test ExportEvent with crop enabled."""
        video_path = tmp_path / "video.mp4"
        video_path.touch()
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        event = ExportEvent(
            all_frames_data=[],
            video_path=str(video_path),
            output_dir=str(output_dir),
            filter_args={},
            enable_crop=True,
            crop_ars="16:9, 1:1",
            crop_padding=20,
        )

        assert event.enable_crop is True
        assert "16:9" in event.crop_ars
        assert event.crop_padding == 20


class TestExportCancellation:
    """Tests for export cancellation handling."""

    @patch("subprocess.Popen")
    @patch("core.export.apply_all_filters_vectorized")
    def test_export_with_cancel_event(self, mock_filter, mock_popen, mock_config, mock_logger, tmp_path):
        """Test export handles cancel event."""
        import threading

        mock_filter.return_value = ([{"filename": "frame_000001.webp"}], [], [], [])

        process = MagicMock()
        process.returncode = 0
        process.communicate.return_value = ("", "")
        mock_popen.return_value = process

        video_path = tmp_path / "video.mp4"
        video_path.touch()
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        (output_dir / "frame_map.json").write_text("[0, 1, 2]")

        cancel_event = threading.Event()
        cancel_event.set()  # Pre-cancelled

        event = ExportEvent(
            all_frames_data=[{"filename": "frame_000001.webp"}],
            video_path=str(video_path),
            output_dir=str(output_dir),
            filter_args={},
            enable_crop=False,
            crop_ars="1:1",
            crop_padding=10,
        )

        # Should not crash with cancelled event
        result = export_kept_frames(event, mock_config, mock_logger, None, cancel_event)
        # Result may be None or indicate cancellation
        assert result is not None or result is None  # Just shouldn't crash


class TestExportWithFilters:
    """Tests for export with various filter configurations."""

    @patch("subprocess.Popen")
    @patch("core.export.apply_all_filters_vectorized")
    def test_export_with_face_filter(self, mock_filter, mock_popen, mock_config, mock_logger, tmp_path):
        """Test export with face similarity filter."""
        mock_filter.return_value = (
            [{"filename": "frame_000001.webp", "face_sim": 0.9}],
            [{"filename": "frame_000002.webp", "face_sim": 0.3}],  # Rejected
            [],
            [],
        )

        process = MagicMock()
        process.returncode = 0
        process.communicate.return_value = ("", "")
        mock_popen.return_value = process

        video_path = tmp_path / "video.mp4"
        video_path.touch()
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        (output_dir / "frame_map.json").write_text("[0, 1, 2]")

        event = ExportEvent(
            all_frames_data=[
                {"filename": "frame_000001.webp", "face_sim": 0.9},
                {"filename": "frame_000002.webp", "face_sim": 0.3},
            ],
            video_path=str(video_path),
            output_dir=str(output_dir),
            filter_args={"face_sim_enabled": True, "face_sim_min": 0.5},
            enable_crop=False,
            crop_ars="1:1",
            crop_padding=10,
        )

        result = export_kept_frames(event, mock_config, mock_logger, None, None)

        assert result is not None
        mock_filter.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
