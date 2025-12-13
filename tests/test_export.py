import pytest
from unittest.mock import MagicMock, patch
import sys
import os
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.events import ExportEvent
from core.export import export_kept_frames

@pytest.fixture
def mock_config():
    return MagicMock()

@pytest.fixture
def mock_logger():
    return MagicMock()

@patch('subprocess.Popen')
@patch('core.export.apply_all_filters_vectorized')
def test_export_kept_frames(mock_filter, mock_popen, mock_config, mock_logger, tmp_path):
    # Setup mocks
    mock_filter.return_value = ([{'filename': 'frame_000001.webp'}], [], [], [])

    process = MagicMock()
    process.returncode = 0
    process.communicate.return_value = ('', '')
    mock_popen.return_value = process

    video_path = tmp_path / "video.mp4"
    video_path.touch()
    output_dir = tmp_path / "out"
    output_dir.mkdir()

    # Create frame_map.json
    (output_dir / "frame_map.json").write_text("[0, 1, 2]")

    event = ExportEvent(
        all_frames_data=[{'filename': 'frame_000001.webp'}],
        video_path=str(video_path),
        output_dir=str(output_dir),
        filter_args={},
        enable_crop=False,
        crop_ars="1:1",
        crop_padding=10
    )

    result = export_kept_frames(event, mock_config, mock_logger, None, None)

    assert "Exported 1 frames" in result
    mock_popen.assert_called()
