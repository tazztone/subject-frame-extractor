from unittest.mock import patch

from core.io_utils import detect_hwaccel


class TestQuickWins:
    # --- io_utils.py ---
    @patch("subprocess.run")
    def test_detect_hwaccel_handles_permission_error(self, mock_run, mock_logger):
        """Test detect_hwaccel handles permission errors from subprocess."""
        mock_run.side_effect = PermissionError("ffmpeg not executable")

        # Should not crash, just return (None, None) or handle error
        encoder, decoder = detect_hwaccel(mock_logger)
        assert encoder is None
        assert decoder is None
        # Check if warning was logged
        assert mock_logger.warning.called
