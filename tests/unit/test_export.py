from unittest.mock import MagicMock, patch

import numpy as np

from core.events import ExportEvent
from core.export import (
    _crop_exported_frames,
    export_kept_frames,
)


class TestExportAdvanced:
    @patch("core.export.cv2.imread")
    @patch("core.export.cv2.imwrite")
    @patch("core.operators.crop.cv2.findContours")
    @patch("core.operators.crop.cv2.boundingRect")
    def test_crop_exported_frames_logic(
        self, mock_boundingRect, mock_findContours, mock_imwrite, mock_imread, tmp_path
    ):
        """Test the logic of cropping exported frames."""
        export_dir = tmp_path / "export"
        export_dir.mkdir()
        masks_root = tmp_path / "masks"
        masks_root.mkdir()

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
        mock_boundingRect.return_value = (10, 10, 20, 20)

        logger = MagicMock()
        cancel_event = MagicMock()
        cancel_event.is_set.return_value = False

        num_cropped = _crop_exported_frames(kept_frames, export_dir, "1:1", 0, masks_root, logger, cancel_event)
        assert num_cropped == 1
        mock_imwrite.assert_called()

    def test_export_kept_frames_no_data(self):
        logger = MagicMock()
        # ExportEvent needs video_path, enable_crop, crop_ars, crop_padding, filter_args
        event = ExportEvent(
            all_frames_data=[],
            output_dir="out",
            video_path="v.mp4",
            enable_crop=False,
            crop_ars="",
            crop_padding=0,
            filter_args={},
        )
        res = export_kept_frames(event, MagicMock(), logger)
        assert "No metadata" in res
