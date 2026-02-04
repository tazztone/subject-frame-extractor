
import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import numpy as np
import cv2
from core.export import _crop_exported_frames

class TestExportAdvanced:

    @patch("cv2.imread")
    @patch("cv2.imwrite")
    @patch("cv2.findContours")
    @patch("cv2.boundingRect")
    def test_crop_exported_frames_logic(self, mock_boundingRect, mock_findContours, mock_imwrite, mock_imread, tmp_path):
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
            np.zeros((100, 100, 3), dtype=np.uint8), # Frame
            np.zeros((100, 100), dtype=np.uint8)     # Mask
        ]
        mock_findContours.return_value = ([np.array([[[10, 10]], [[20, 20]]])], None)
        mock_boundingRect.return_value = (10, 10, 20, 20) # x, y, w, h

        logger = MagicMock()
        cancel_event = MagicMock()
        cancel_event.is_set.return_value = False

        # Test 1:Basic Crop
        num_cropped = _crop_exported_frames(
            kept_frames, export_dir, "1:1", 0, masks_root, logger, cancel_event
        )
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
        masks_root = tmp_path / "masks" # Exists but empty files
        masks_root.mkdir()

        kept_frames = [{"filename": "frame_000001.png", "mask_path": "mask_000001.png"}]
        logger = MagicMock()
        cancel_event = MagicMock()
        cancel_event.is_set.return_value = False

        num_cropped = _crop_exported_frames(
            kept_frames, export_dir, "1:1", 0, masks_root, logger, cancel_event
        )
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

        mock_imread.side_effect = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((100, 100), dtype=np.uint8)
        ]
        # No contours found
        mock_findContours.return_value = ([], None)

        logger = MagicMock()
        cancel_event = MagicMock()
        cancel_event.is_set.return_value = False

        num_cropped = _crop_exported_frames(
            kept_frames, export_dir, "1:1", 0, masks_root, logger, cancel_event
        )
        assert num_cropped == 0
