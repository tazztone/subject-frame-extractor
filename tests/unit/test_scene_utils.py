"""
Tests for scene utilities - SeedSelector, MaskPropagator, SubjectMasker.

Uses fixtures from conftest.py for mock setup.
"""

import threading
from queue import Queue
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from core.models import Scene
from core.scene_utils import MaskPropagator, SeedSelector, SubjectMasker


# Helper to create a tensor mock that supports comparison operations
def create_tensor_mock(shape=(100, 100), val=1.0):
    mock = MagicMock()
    mock.__gt__ = MagicMock(
        return_value=MagicMock(
            cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.ones(shape, dtype=bool))))
        )
    )
    mock.ndim = len(shape)
    return mock


class TestSeedSelector:
    @pytest.fixture
    def selector(self, mock_config_simple, mock_logger, mock_params):
        tracker = MagicMock()
        face_analyzer = MagicMock()
        return SeedSelector(mock_params, mock_config_simple, face_analyzer, None, tracker, mock_logger)

    def test_select_seed_strategies(self, selector):
        # Setup detections
        d1 = {"bbox": [0, 0, 40, 40], "conf": 0.5, "type": "person"}
        d2 = {"bbox": [40, 40, 60, 60], "conf": 0.95, "type": "person"}
        d3 = {"bbox": [80, 0, 90, 100], "conf": 0.8, "type": "person"}
        detections = [d1, d2, d3]
        selector.tracker.detect_objects.return_value = detections
        frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)

        selector.params.seed_strategy = "Largest Person"
        bbox, _ = selector.select_seed(frame_rgb)
        assert bbox == [0, 0, 42, 42]

        selector.params.seed_strategy = "Highest Confidence"
        bbox, _ = selector.select_seed(frame_rgb)
        assert bbox == [39, 39, 22, 22]

        selector.params.seed_strategy = "Center-most Person"
        bbox, _ = selector.select_seed(frame_rgb)
        assert bbox == [39, 39, 22, 22]

        selector.params.seed_strategy = "Tallest Person"
        bbox, _ = selector.select_seed(frame_rgb)
        assert bbox == [79, 0, 11, 99]

    def test_identity_first_seed(self, selector):
        selector.params.primary_seed_strategy = "👤 By Face"
        selector.params.enable_face_filter = True
        selector.reference_embedding = np.ones(128)

        mock_face = MagicMock()
        mock_face.normed_embedding = np.ones(128)
        mock_face.bbox = np.array([10, 10, 30, 30])
        selector.face_analyzer.get.return_value = [mock_face]

        selector.tracker.detect_objects.return_value = [{"bbox": [5, 5, 35, 35], "conf": 0.9, "type": "person"}]
        selector.config.seeding_face_similarity_threshold = 0.5

        frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox, details = selector.select_seed(frame_rgb)

        assert bbox is not None
        assert details.get("face_contained") is True

    def test_face_with_text_fallback_seed_success(self, selector):
        selector.params.primary_seed_strategy = "🔄 Face + Text Fallback"
        selector.reference_embedding = np.ones(128)
        with patch.object(selector, "_identity_first_seed") as mock_face_seed:
            mock_face_seed.return_value = ([0, 0, 10, 10], {})
            frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
            bbox, _ = selector.select_seed(frame_rgb)
            assert bbox == [0, 0, 10, 10]
            mock_face_seed.assert_called_once()

    def test_face_with_text_fallback_seed_fallback(self, selector):
        selector.params.primary_seed_strategy = "🔄 Face + Text Fallback"
        selector.reference_embedding = np.ones(128)
        with patch.object(selector, "_identity_first_seed", return_value=(None, {})):
            with patch.object(selector, "_object_first_seed") as mock_obj_seed:
                mock_obj_seed.return_value = ([10, 10, 20, 20], {})
                frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
                bbox, _ = selector.select_seed(frame_rgb)
                assert bbox == [10, 10, 20, 20]
                mock_obj_seed.assert_called_once()

    @patch("core.scene_utils.seed_selector.postprocess_mask")
    def test_get_mask_for_bbox_success(self, mock_post, selector, tmp_path):
        frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = [0, 0, 50, 50]

        # Mock tracker returning a simple numpy array for mask
        selector.tracker.add_bbox_prompt.return_value = np.ones((100, 100), dtype=bool)
        mock_post.side_effect = lambda x, **k: x  # identity

        with patch("core.scene_utils.seed_selector.rgb_to_pil") as mock_pil:
            mock_img = MagicMock()
            mock_pil.return_value = mock_img

            mask = selector._get_mask_for_bbox(frame_rgb, bbox)

            assert mask is not None
            selector.tracker.init_video.assert_called()
            selector.tracker.add_bbox_prompt.assert_called()

    def test_get_mask_for_bbox_error(self, selector):
        frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = [0, 0, 50, 50]

        selector.tracker.init_video.side_effect = RuntimeError("SAM Error")

        # Patch BaseException check issue if needed, but RuntimeError is standard
        # However, pytest's MagicMock might be tricky if not careful
        # Here we just expect None return

        # If the code catches (torch.cuda.OutOfMemoryError, RuntimeError), we need to ensure torch is mocked correctly
        # In conftest, torch is a MagicMock. accessing torch.cuda.OutOfMemoryError might return a MagicMock which is not a type.
        # We need to ensure torch.cuda.OutOfMemoryError is a valid exception type for except clause.

        with patch("torch.cuda.OutOfMemoryError", RuntimeError):
            mask = selector._get_mask_for_bbox(frame_rgb, bbox)
            assert mask is None


class TestMaskPropagator:
    @patch("core.scene_utils.mask_propagator.postprocess_mask", side_effect=lambda x, **k: x)
    def test_propagate_success(self, mock_post, mock_config_simple, mock_logger, mock_params):
        tracker = MagicMock()
        tracker.init_video.return_value = None
        tracker.add_bbox_prompt.return_value = np.ones((100, 100), dtype=bool)
        tracker.propagate.return_value = []

        propagator = MaskPropagator(mock_params, tracker, threading.Event(), Queue(), mock_config_simple, mock_logger)
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        masks, areas, empties, errors = propagator.propagate(frames, 0, [0, 0, 10, 10])

        assert len(masks) == 1
        assert masks[0] is not None

    def test_propagate_cancel(self, mock_config_simple, mock_logger, mock_params):
        tracker = MagicMock()
        cancel_event = threading.Event()
        cancel_event.set()

        # Mock tracker methods so it doesn't fail before checking cancel
        tracker.init_video.return_value = None
        tracker.add_bbox_prompt.return_value = np.ones((100, 100), dtype=bool)

        propagator = MaskPropagator(mock_params, tracker, cancel_event, Queue(), mock_config_simple, mock_logger)
        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]

        # Mock postprocess_mask to avoid TypeError
        with patch("core.scene_utils.mask_propagator.postprocess_mask", side_effect=lambda x, **k: x):
            res = propagator.propagate(frames, 0, [0, 0, 10, 10])
            assert res is not None
            assert len(res[0]) == 1  # masks


class TestSubjectMasker:
    @patch("core.scene_utils.subject_masker.create_frame_map", return_value={0: "frame_0.png"})
    def test_run_propagation(self, mock_create_map, mock_config_simple, mock_logger, mock_params, tmp_path):
        mock_model_registry = MagicMock()
        mock_tracker = MagicMock()
        mock_model_registry.get_tracker.return_value = mock_tracker

        with patch("core.scene_utils.subject_masker.MaskPropagator") as MockPropagator:
            instance = MockPropagator.return_value
            instance.propagate_video.return_value = (
                {0: np.ones((10, 10), dtype=np.uint8)}, {0: 100.0}, {0: False}, {0: None}
            )

            masker = SubjectMasker(
                mock_params,
                Queue(),
                threading.Event(),
                mock_config_simple,
                logger=mock_logger,
                model_registry=mock_model_registry,
                thumbnail_manager=MagicMock()
            )
            masker.frame_map = {0: "frame_0.png"}

            scene = Scene(
                shot_id=1, start_frame=0, end_frame=1, best_frame=0, seed_result={"bbox": [0, 0, 10, 10], "details": {}}
            )

            with patch("core.scene_utils.subject_masker.SubjectMasker._load_shot_frames") as mock_load:
                mock_load.return_value = [(0, np.zeros((10, 10, 3), dtype=np.uint8), (10, 10))]
                frames_dir = tmp_path / "frames"
                frames_dir.mkdir()
                (frames_dir / "video_lowres.mp4").touch()
                result = masker.run_propagation(str(frames_dir), [scene])
                assert result

    def test_load_shot_frames(self, mock_config_simple, mock_logger, mock_params, tmp_path):
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        img_path = frames_dir / "frame_00000.png"

        # Write real image
        cv2.imwrite(str(img_path), np.zeros((10, 10, 3), dtype=np.uint8))

        masker = SubjectMasker(
            mock_params, Queue(), threading.Event(), mock_config_simple, logger=mock_logger, model_registry=MagicMock()
        )
        masker.frame_map = {0: "frame_00000.png"}

        # Mock thumbnail manager to return something valid
        masker.thumbnail_manager = MagicMock()
        masker.thumbnail_manager.get.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

        thumb_dir = tmp_path / "thumbs"
        thumb_dir.mkdir()

        # Ensure we pass Path object for thumb_dir
        frames = masker._load_shot_frames(str(frames_dir), thumb_dir, 0, 1)  # end is exclusive
        assert len(frames) == 1
        assert frames[0][0] == 0
