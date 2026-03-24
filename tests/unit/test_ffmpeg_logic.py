import subprocess
from unittest.mock import MagicMock, patch

from core.scene_utils.ffmpeg import perform_ffmpeg_export


def test_perform_ffmpeg_export_timeout(tmp_path):
    """Test perform_ffmpeg_export handling of timeout and cleanup."""
    logger = MagicMock()
    video_path = "test.mp4"
    frames = [1, 2, 3]
    export_dir = tmp_path / "export"
    export_dir.mkdir()

    # Create a partial file
    partial_file = export_dir / "frame_000001.png"
    partial_file.touch()

    with patch("subprocess.Popen") as mock_popen:
        process = MagicMock()
        process.communicate.side_effect = subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=1)
        mock_popen.return_value = process

        success, stderr = perform_ffmpeg_export(video_path, frames, export_dir, logger, timeout=1)

        assert success is False
        assert "timed out" in stderr.lower()
        process.kill.assert_called()
        # Verify cleanup of frame_*.png
        assert not partial_file.exists()


def test_perform_ffmpeg_export_failure(tmp_path):
    """Test perform_ffmpeg_export handling of non-zero exit code and cleanup."""
    logger = MagicMock()
    video_path = "test.mp4"
    frames = [1, 2, 3]
    export_dir = tmp_path / "export"
    export_dir.mkdir()

    partial_file = export_dir / "frame_000001.png"
    partial_file.touch()

    with patch("subprocess.Popen") as mock_popen:
        process = MagicMock()
        process.returncode = 1
        process.communicate.return_value = ("", "FFmpeg failed")
        mock_popen.return_value = process

        success, stderr = perform_ffmpeg_export(video_path, frames, export_dir, logger)

        assert success is False
        assert "FFmpeg failed" in stderr
        # Verify cleanup
        assert not partial_file.exists()
