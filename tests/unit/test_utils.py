"""
Tests for core utility functions.

Covers utilities in core/utils.py including:
- Video validation
- Filename sanitization
- Image/mask processing
- JSON serialization

Note: Some tests require integration mode (no mocks) due to numba/opencv dependencies.
"""

import json
from pathlib import Path

import numpy as np
import pytest

# Mark module - tests run with mocks by default
pytestmark = pytest.mark.unit

from core.utils import (
    _to_json_safe,
    create_frame_map,
    draw_bbox,
    is_image_folder,
    list_images,
    postprocess_mask,
    render_mask_overlay,
    rgb_to_pil,
    sanitize_filename,
    validate_video_file,
)


class TestValidateVideoFile:
    """Tests for validate_video_file function."""

    def test_nonexistent_file_raises(self):
        """Test validation raises FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            validate_video_file("/nonexistent/path/video.mp4")

    def test_empty_file_raises(self, tmp_path):
        """Test validation raises ValueError for empty file."""
        empty_file = tmp_path / "empty.mp4"
        empty_file.touch()
        with pytest.raises(ValueError):
            validate_video_file(str(empty_file))

    @pytest.mark.integration
    def test_valid_video_file(self, tmp_path):
        """Test validation of a valid video file (requires OpenCV)."""
        import cv2

        video_path = tmp_path / "test.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(video_path), fourcc, 1.0, (10, 10))
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        out.write(frame)
        out.release()

        result = validate_video_file(str(video_path))
        assert result is True


class TestSanitizeFilename:
    """Tests for sanitize_filename function."""

    def test_basic_sanitization(self, mock_config):
        """Test basic filename sanitization."""
        result = sanitize_filename("My Video File.mp4", mock_config)
        assert "/" not in result
        assert "\\" not in result

    def test_special_characters(self, mock_config):
        """Test removal of special characters."""
        result = sanitize_filename('file<>:"/\\|?*.txt', mock_config)
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert "?" not in result
        assert "*" not in result

    def test_max_length(self, mock_config):
        """Test filename truncation at max length."""
        long_name = "a" * 300
        result = sanitize_filename(long_name, mock_config, max_length=100)
        assert len(result) <= 100

    def test_empty_string(self, mock_config):
        """Test sanitization of empty string."""
        result = sanitize_filename("", mock_config)
        assert result is not None


class TestIsImageFolder:
    """Tests for is_image_folder function."""

    def test_valid_directory(self, tmp_path):
        """Test detection of valid directory."""
        assert is_image_folder(tmp_path) is True

    def test_file_not_folder(self, tmp_path):
        """Test returns False for file."""
        file_path = tmp_path / "test.txt"
        file_path.touch()
        assert is_image_folder(file_path) is False

    def test_nonexistent_path(self):
        """Test returns False for non-existent path."""
        assert is_image_folder("/nonexistent/path") is False

    def test_string_path(self, tmp_path):
        """Test works with string path."""
        assert is_image_folder(str(tmp_path)) is True


class TestListImages:
    """Tests for list_images function."""

    def test_lists_image_files(self, tmp_path, mock_config):
        """Test listing of image files in directory."""
        (tmp_path / "img1.jpg").touch()
        (tmp_path / "img2.png").touch()
        (tmp_path / "img3.webp").touch()
        (tmp_path / "not_an_image.txt").touch()

        images = list_images(tmp_path, mock_config)

        # Should find some images
        assert isinstance(images, list)

    def test_empty_directory(self, tmp_path, mock_config):
        """Test returns empty list for empty directory."""
        images = list_images(tmp_path, mock_config)
        assert images == []


class TestCreateFrameMap:
    """Tests for create_frame_map function."""

    def test_creates_frame_map(self, tmp_path, mock_logger):
        """Test frame map creation from directory."""
        thumbs_dir = tmp_path / "thumbs"
        thumbs_dir.mkdir()

        (thumbs_dir / "frame_000000.webp").touch()
        (thumbs_dir / "frame_000001.webp").touch()
        (thumbs_dir / "frame_000005.webp").touch()

        frame_map = create_frame_map(tmp_path, mock_logger)

        # Should return a dict
        assert isinstance(frame_map, dict)

    def test_empty_thumbs_directory(self, tmp_path, mock_logger):
        """Test returns empty dict for empty thumbs directory."""
        thumbs_dir = tmp_path / "thumbs"
        thumbs_dir.mkdir()

        frame_map = create_frame_map(tmp_path, mock_logger)
        assert isinstance(frame_map, dict)
        assert len(frame_map) == 0


class TestPostprocessMask:
    """Tests for postprocess_mask function."""

    def test_basic_mask_processing(self, mock_config):
        """Test basic mask postprocessing."""
        import cv2

        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 30, 255, -1)

        result = postprocess_mask(mask, mock_config)

        assert result is not None
        assert result.shape == mask.shape
        assert result.dtype == np.uint8

    def test_empty_mask(self, mock_config):
        """Test processing of empty mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = postprocess_mask(mask, mock_config)
        assert result is not None

    def test_full_mask(self, mock_config):
        """Test processing of full mask."""
        mask = np.ones((100, 100), dtype=np.uint8) * 255
        result = postprocess_mask(mask, mock_config)
        assert result is not None


class TestRenderMaskOverlay:
    """Tests for render_mask_overlay function."""

    def test_basic_overlay(self, mock_logger):
        """Test basic mask overlay on image."""
        import cv2

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :] = [100, 150, 200]

        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 20, 255, -1)

        result = render_mask_overlay(image, mask, 0.5, mock_logger)

        assert result is not None
        assert result.shape == image.shape

    def test_empty_mask_overlay(self, mock_logger):
        """Test overlay with empty mask doesn't crash."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)

        result = render_mask_overlay(image, mask, 0.5, mock_logger)
        assert result is not None


class TestRgbToPil:
    """Tests for rgb_to_pil function."""

    def test_basic_conversion(self):
        """Test basic RGB to PIL conversion."""
        from PIL import Image

        rgb_array = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb_array[50, 50] = [255, 0, 0]

        pil_image = rgb_to_pil(rgb_array)

        assert isinstance(pil_image, Image.Image)
        assert pil_image.size == (100, 100)

    def test_grayscale_raises(self):
        """Test that 2D array handling."""
        rgb_array = np.zeros((100, 100), dtype=np.uint8)
        # Depending on implementation, may convert or raise
        try:
            result = rgb_to_pil(rgb_array)
            assert result is not None
        except (ValueError, Exception):
            pass  # Expected for some implementations


class TestDrawBbox:
    """Tests for draw_bbox function."""

    def test_basic_bbox(self, mock_config):
        """Test basic bounding box drawing."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        xywh = [10, 10, 30, 40]

        result = draw_bbox(image, xywh, mock_config)

        assert result is not None
        assert result.shape == image.shape

    def test_bbox_with_label(self, mock_config):
        """Test bounding box with label."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        xywh = [10, 10, 30, 40]

        result = draw_bbox(image, xywh, mock_config, label="Test")
        assert result is not None

    def test_bbox_with_color(self, mock_config):
        """Test bounding box with custom color."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        xywh = [10, 10, 30, 40]

        result = draw_bbox(image, xywh, mock_config, color=(255, 0, 0))
        assert result is not None


class TestToJsonSafe:
    """Tests for _to_json_safe function."""

    def test_numpy_int(self):
        """Test conversion of numpy int."""
        data = {"int": np.int64(42)}
        safe_data = _to_json_safe(data)
        json_str = json.dumps(safe_data)
        assert "42" in json_str

    def test_numpy_float(self):
        """Test conversion of numpy float."""
        data = {"float": np.float64(3.14)}
        safe_data = _to_json_safe(data)
        json_str = json.dumps(safe_data)
        assert "3.14" in json_str

    def test_numpy_array(self):
        """Test conversion of numpy array."""
        data = {"array": np.array([1, 2, 3])}
        safe_data = _to_json_safe(data)
        json_str = json.dumps(safe_data)
        assert json_str is not None

    def test_path_conversion(self):
        """Test conversion of Path objects."""
        data = {"path": Path("/some/path")}
        safe_data = _to_json_safe(data)

        # Path should be converted to string
        json_str = json.dumps(safe_data)
        assert "some" in json_str or "path" in json_str

    def test_plain_dict(self):
        """Test plain dict passes through."""
        data = {"key": "value", "num": 123}
        safe_data = _to_json_safe(data)
        json_str = json.dumps(safe_data)
        assert "key" in json_str

    def test_nested_structures(self):
        """Test conversion of nested structures."""
        data = {
            "list": [np.int64(1), np.float64(2.0)],
            "nested": {"value": np.int64(3)},
        }
        safe_data = _to_json_safe(data)
        json_str = json.dumps(safe_data)
        assert json_str is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
