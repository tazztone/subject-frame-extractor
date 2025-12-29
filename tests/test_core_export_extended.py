
import os
import shutil
import tempfile
import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np
import cv2

from core.events import ExportEvent
from core.export import export_kept_frames, dry_run_export
from core.config import Config

class TestExportExtended:
    """
    Extended tests for core/export.py to improve coverage.
    """

    @pytest.fixture
    def mock_components(self):
        config = Config()
        logger = MagicMock()
        thumb_manager = MagicMock()
        cancel_event = MagicMock()
        cancel_event.is_set.return_value = False
        return config, logger, thumb_manager, cancel_event

    @pytest.fixture
    def sample_data(self, tmp_path):
        """Creates sample frame data and directories."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        thumbs_dir = output_dir / "thumbs"
        thumbs_dir.mkdir()

        # Create a dummy video file
        video_path = tmp_path / "video.mp4"
        video_path.touch()

        frames = []
        frame_map = []
        for i in range(5):
            filename = f"frame_{i+1:06d}.jpg"
            (thumbs_dir / filename).touch()
            frames.append({
                "filename": filename,
                "frame_num": i,
                "timestamp": i * 1.0,
                "phash": "0000000000000000",
                "kept": True, # All kept initially
                "metrics": {"quality_score": 10.0 + i} # 10, 11, 12, 13, 14
            })
            frame_map.append(i * 10) # Original frame numbers

        with open(output_dir / "frame_map.json", "w") as f:
            json.dump(frame_map, f)

        return output_dir, video_path, frames

    def test_export_kept_frames_basic(self, mock_components, sample_data):
        """Test basic export functionality without cropping."""
        config, logger, thumb_manager, cancel_event = mock_components
        output_dir, video_path, frames = sample_data

        with patch("cv2.imread", return_value=np.zeros((100, 100, 3), dtype=np.uint8)), \
             patch("cv2.imwrite") as mock_write, \
             patch("core.export._perform_ffmpeg_export", return_value=(True, None)) as mock_ffmpeg, \
             patch("core.export._rename_exported_frames") as mock_rename:

            event = ExportEvent(
                all_frames_data=frames,
                output_dir=str(output_dir),
                video_path=str(video_path),
                enable_crop=False,
                crop_ars="",
                crop_padding=0,
                filter_args={}
            )

            result = export_kept_frames(event, config, logger, thumb_manager, cancel_event)

            assert "Exported 5 frames" in result
            assert mock_ffmpeg.called
            assert mock_rename.called

    def test_export_kept_frames_with_crop(self, mock_components, sample_data):
        """Test export with cropping enabled."""
        config, logger, thumb_manager, cancel_event = mock_components
        output_dir, video_path, frames = sample_data

        with patch("cv2.imread", return_value=np.zeros((100, 100, 3), dtype=np.uint8)), \
             patch("cv2.imwrite") as mock_write, \
             patch("core.export._perform_ffmpeg_export", return_value=(True, None)), \
             patch("core.export._rename_exported_frames"), \
             patch("core.export._crop_exported_frames", return_value=5) as mock_crop:

            event = ExportEvent(
                all_frames_data=frames,
                output_dir=str(output_dir),
                video_path=str(video_path),
                enable_crop=True,
                crop_ars="1:1",
                crop_padding=10,
                filter_args={}
            )

            result = export_kept_frames(event, config, logger, thumb_manager, cancel_event)

            assert "Exported 5 frames" in result
            assert mock_crop.called

    def test_dry_run_export(self, mock_components, sample_data):
        """Test dry run export with filtering."""
        config, _, _, _ = mock_components
        output_dir, video_path, frames = sample_data

        # Apply filter: keep only quality_score > 12.0
        # Frames have scores: 10, 11, 12, 13, 14.
        # > 12 means 13 and 14 kept? (2 frames)
        # Or >=? Let's use min=12.5 to keep 13, 14 (2 frames).

        event = ExportEvent(
            all_frames_data=frames,
            output_dir=str(output_dir),
            video_path=str(video_path),
            enable_crop=False,
            crop_ars="",
            crop_padding=0,
            filter_args={"quality_score_min": 12.5}
        )

        result = dry_run_export(event, config)

        assert "Dry Run: 2 frames" in result
        assert "FFmpeg command" in result
        assert re.search(r"ffmpeg .*-vf select=", result)

    def test_export_empty_frames(self, mock_components, sample_data):
        """Test export with no frames."""
        config, logger, thumb_manager, cancel_event = mock_components
        output_dir, video_path, _ = sample_data

        event = ExportEvent(
            all_frames_data=[],
            output_dir=str(output_dir),
            video_path=str(video_path),
            enable_crop=False,
            crop_ars="",
            crop_padding=0,
            filter_args={}
        )

        result = export_kept_frames(event, config, logger, thumb_manager, cancel_event)
        assert "No metadata to export" in result

    def test_export_no_video_path(self, mock_components, sample_data):
        """Test export with missing video path."""
        config, logger, thumb_manager, cancel_event = mock_components
        output_dir, _, frames = sample_data

        event = ExportEvent(
            all_frames_data=frames,
            output_dir=str(output_dir),
            video_path="",
            enable_crop=False,
            crop_ars="",
            crop_padding=0,
            filter_args={}
        )

        result = export_kept_frames(event, config, logger, thumb_manager, cancel_event)
        assert "[ERROR] Original video path is required" in result
