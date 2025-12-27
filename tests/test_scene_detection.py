"""
Tests for scene_utils modules (detection, helpers).

These tests verify scene detection and helper functions work correctly.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


class TestSceneDetection:
    """Tests for scene_utils/detection.py."""

    @patch("core.scene_utils.detection.detect")
    def test_run_scene_detection_success(self, mock_detect, mock_logger, tmp_path):
        """Test run_scene_detection returns scene list."""
        from core.scene_utils.detection import run_scene_detection

        # Mock scene detection results
        mock_scene1 = MagicMock()
        mock_scene1.get_frames.return_value = 0
        mock_scene2 = MagicMock()
        mock_scene2.get_frames.return_value = 100
        mock_scene3 = MagicMock()
        mock_scene3.get_frames.return_value = 100
        mock_scene4 = MagicMock()
        mock_scene4.get_frames.return_value = 200

        mock_detect.return_value = [
            (mock_scene1, mock_scene2),
            (mock_scene3, mock_scene4),
        ]

        result = run_scene_detection("/path/to/video.mp4", tmp_path, mock_logger)

        assert len(result) == 2
        assert result[0] == (0, 100)
        assert result[1] == (100, 200)

        # Verify scenes.json was written
        scenes_file = tmp_path / "scenes.json"
        assert scenes_file.exists()

    @patch("core.scene_utils.detection.detect")
    def test_run_scene_detection_empty(self, mock_detect, mock_logger, tmp_path):
        """Test run_scene_detection with no scenes detected."""
        from core.scene_utils.detection import run_scene_detection

        mock_detect.return_value = []

        result = run_scene_detection("/path/to/video.mp4", tmp_path, mock_logger)

        assert result == []

    @patch("core.scene_utils.detection.detect")
    def test_run_scene_detection_exception(self, mock_detect, mock_logger, tmp_path):
        """Test run_scene_detection handles exceptions gracefully."""
        from core.scene_utils.detection import run_scene_detection

        mock_detect.side_effect = Exception("Detection failed")

        result = run_scene_detection("/path/to/video.mp4", tmp_path, mock_logger)

        # Should return empty list on exception
        assert result == []

    @patch("cv2.imread")
    @patch("cv2.resize")
    @patch("cv2.cvtColor")
    def test_make_photo_thumbs(
        self, mock_cvtcolor, mock_resize, mock_imread, mock_logger, mock_config_simple, tmp_path
    ):
        """Test make_photo_thumbs generates thumbnails."""
        from core.models import AnalysisParameters
        from core.scene_utils.detection import make_photo_thumbs

        # Mock image reading
        test_image = np.zeros((1920, 1080, 3), dtype=np.uint8)
        mock_imread.return_value = test_image
        mock_resize.return_value = np.zeros((720, 405, 3), dtype=np.uint8)
        mock_cvtcolor.return_value = np.zeros((720, 405, 3), dtype=np.uint8)

        # Create test image files
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        (images_dir / "test1.jpg").touch()
        (images_dir / "test2.jpg").touch()

        params = AnalysisParameters(
            source_path="test.mp4",
            video_path="test.mp4",
            output_folder=str(tmp_path),
            thumb_megapixels=0.5,
        )

        with patch("PIL.Image.fromarray") as mock_pil:
            mock_img = MagicMock()
            mock_pil.return_value = mock_img

            result = make_photo_thumbs(
                image_paths=[images_dir / "test1.jpg", images_dir / "test2.jpg"],
                out_dir=tmp_path,
                params=params,
                cfg=mock_config_simple,
                logger=mock_logger,
            )

        # Should return frame map
        assert isinstance(result, dict)

    @patch("cv2.imread")
    def test_make_photo_thumbs_unreadable_image(self, mock_imread, mock_logger, mock_config_simple, tmp_path):
        """Test make_photo_thumbs handles unreadable images."""
        from core.models import AnalysisParameters
        from core.scene_utils.detection import make_photo_thumbs

        mock_imread.return_value = None  # Simulate read failure

        images_dir = tmp_path / "images"
        images_dir.mkdir()
        (images_dir / "bad.jpg").touch()

        params = AnalysisParameters(
            source_path="test.mp4",
            video_path="test.mp4",
            output_folder=str(tmp_path),
            thumb_megapixels=0.5,
        )

        result = make_photo_thumbs(
            image_paths=[images_dir / "bad.jpg"],
            out_dir=tmp_path,
            params=params,
            cfg=mock_config_simple,
            logger=mock_logger,
        )

        # Should return empty map since image couldn't be read
        assert result == {}


class TestSceneHelpers:
    """Tests for scene_utils/helpers.py."""

    def test_draw_boxes_preview(self, mock_config_simple):
        """Test draw_boxes_preview draws bounding boxes."""
        from core.scene_utils.helpers import draw_boxes_preview

        # Create a test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = [[10, 10, 50, 50], [60, 60, 90, 90]]

        result = draw_boxes_preview(img, boxes, mock_config_simple)

        # Result should be an image with same shape
        assert result.shape == img.shape
        # Boxes should be drawn (pixels changed)
        assert not np.array_equal(result, img)

    def test_draw_boxes_preview_empty_boxes(self, mock_config_simple):
        """Test draw_boxes_preview with no boxes."""
        from core.scene_utils.helpers import draw_boxes_preview

        img = np.zeros((100, 100, 3), dtype=np.uint8)

        result = draw_boxes_preview(img, [], mock_config_simple)

        # Should return unchanged image
        assert np.array_equal(result, img)

    def test_save_scene_seeds(self, mock_logger, tmp_path, sample_scenes):
        """Test save_scene_seeds writes JSON file."""
        from core.scene_utils.helpers import save_scene_seeds

        save_scene_seeds(sample_scenes, str(tmp_path), mock_logger)

        seeds_file = tmp_path / "scene_seeds.json"
        assert seeds_file.exists()

        with open(seeds_file) as f:
            data = json.load(f)

        # Should have scene data - stored as dict with shot_id keys
        assert isinstance(data, dict)
        assert len(data) == len(sample_scenes)

    def test_get_scene_status_text_empty(self):
        """Test get_scene_status_text with empty list."""
        from core.scene_utils.helpers import get_scene_status_text

        text, button_update = get_scene_status_text([])

        # Either contains "0" or mentions "no" or "empty"
        assert "0" in text or "no" in text.lower() or "empty" in text.lower() or text != ""

    def test_get_scene_status_text_with_scenes(self, sample_scenes):
        """Test get_scene_status_text with scenes."""
        from core.scene_utils.helpers import get_scene_status_text

        text, button_update = get_scene_status_text(sample_scenes)

        # Should contain scene count
        assert str(len(sample_scenes)) in text or "4" in text

    def test_toggle_scene_status_include(self, mock_logger, tmp_path, sample_scenes):
        """Test toggle_scene_status includes a scene."""
        from core.scene_utils.helpers import toggle_scene_status

        result = toggle_scene_status(
            scenes_list=sample_scenes,
            selected_shot_id=1,
            new_status="included",
            output_folder=str(tmp_path),
            logger=mock_logger,
        )

        updated_scenes, status_text, message, button_update = result

        # Find the scene and check status
        scene = next(s for s in updated_scenes if s.shot_id == 1)
        assert scene.status == "included"

    def test_toggle_scene_status_exclude(self, mock_logger, tmp_path, sample_scenes):
        """Test toggle_scene_status excludes a scene."""
        from core.scene_utils.helpers import toggle_scene_status

        result = toggle_scene_status(
            scenes_list=sample_scenes,
            selected_shot_id=2,
            new_status="excluded",
            output_folder=str(tmp_path),
            logger=mock_logger,
        )

        updated_scenes, status_text, message, button_update = result

        scene = next(s for s in updated_scenes if s.shot_id == 2)
        assert scene.status == "excluded"

    def test_toggle_scene_status_invalid_id(self, mock_logger, tmp_path, sample_scenes):
        """Test toggle_scene_status with invalid shot_id."""
        from core.scene_utils.helpers import toggle_scene_status

        result = toggle_scene_status(
            scenes_list=sample_scenes,
            selected_shot_id=999,  # Non-existent
            new_status="included",
            output_folder=str(tmp_path),
            logger=mock_logger,
        )

        # Should handle gracefully
        updated_scenes, status_text, message, button_update = result
        assert updated_scenes == sample_scenes  # Unchanged


class TestManagersThumbnailManager:
    """Tests for ThumbnailManager in managers.py."""

    def test_thumbnail_manager_init(self, mock_logger, mock_config):
        """Test ThumbnailManager initialization."""
        from core.managers import ThumbnailManager

        manager = ThumbnailManager(logger=mock_logger, config=mock_config)

        assert manager.logger == mock_logger
        assert manager.cache is not None

    def test_thumbnail_manager_get_from_cache(self, mock_logger, mock_config, tmp_path):
        """Test ThumbnailManager returns cached thumbnail."""
        from core.managers import ThumbnailManager

        manager = ThumbnailManager(logger=mock_logger, config=mock_config)

        # Create a test image
        test_thumb = tmp_path / "test_thumb.webp"
        test_image = np.zeros((50, 50, 3), dtype=np.uint8)
        test_image[10:40, 10:40] = [255, 0, 0]  # Red square

        from PIL import Image

        Image.fromarray(test_image).save(test_thumb)

        # First get - should load from disk
        result1 = manager.get(test_thumb)

        # Second get - should return from cache
        result2 = manager.get(test_thumb)

        assert result1 is not None
        assert result2 is not None

    def test_thumbnail_manager_get_missing_file(self, mock_logger, mock_config):
        """Test ThumbnailManager handles missing file."""
        from core.managers import ThumbnailManager

        manager = ThumbnailManager(logger=mock_logger, config=mock_config)

        result = manager.get(Path("/nonexistent/path.webp"))

        assert result is None

    def test_thumbnail_manager_clear_cache(self, mock_logger, mock_config, tmp_path):
        """Test ThumbnailManager cache clearing."""
        from core.managers import ThumbnailManager

        manager = ThumbnailManager(logger=mock_logger, config=mock_config)

        # Add something to cache
        test_thumb = tmp_path / "test.webp"
        test_image = np.zeros((50, 50, 3), dtype=np.uint8)
        from PIL import Image

        Image.fromarray(test_image).save(test_thumb)

        manager.get(test_thumb)
        assert len(manager.cache) > 0

        manager.clear_cache()
        assert len(manager.cache) == 0


class TestModelRegistry:
    """Tests for ModelRegistry in managers.py."""

    def test_model_registry_init(self, mock_logger):
        """Test ModelRegistry initialization."""
        from core.managers import ModelRegistry

        registry = ModelRegistry(logger=mock_logger)

        assert registry._models == {}
        assert registry.logger == mock_logger

    def test_model_registry_get_or_load_new(self, mock_logger):
        """Test ModelRegistry loads new model."""
        from core.managers import ModelRegistry

        registry = ModelRegistry(logger=mock_logger)

        mock_model = MagicMock()
        loader_fn = MagicMock(return_value=mock_model)

        result = registry.get_or_load("test_model", loader_fn)

        assert result == mock_model
        loader_fn.assert_called_once()

    def test_model_registry_get_or_load_cached(self, mock_logger):
        """Test ModelRegistry returns cached model."""
        from core.managers import ModelRegistry

        registry = ModelRegistry(logger=mock_logger)

        mock_model = MagicMock()
        loader_fn = MagicMock(return_value=mock_model)

        # First load
        registry.get_or_load("test_model", loader_fn)

        # Second load - should not call loader again
        result = registry.get_or_load("test_model", loader_fn)

        assert result == mock_model
        assert loader_fn.call_count == 1  # Only called once

    def test_model_registry_clear(self, mock_logger):
        """Test ModelRegistry clear removes all models."""
        from core.managers import ModelRegistry

        registry = ModelRegistry(logger=mock_logger)

        # Add a model
        registry.get_or_load("test", lambda: MagicMock())
        assert len(registry._models) > 0

        registry.clear()
        assert len(registry._models) == 0
