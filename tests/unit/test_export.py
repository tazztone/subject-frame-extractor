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
