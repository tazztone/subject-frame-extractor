from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from core.models import AnalysisParameters, Scene
from core.scene_utils.subject_masker import SubjectMasker


class TestSubjectMaskerCoverage:
    """
    Comprehensive coverage tests for core/scene_utils/subject_masker.py
    Focuses on run_propagation loop, resource cleanup, and edge cases.
    """

    @pytest.fixture
    def mock_deps(self):
        config = MagicMock()
        config.retry_max_attempts = 1
        config.retry_backoff_seconds = [1]
        config.models_dir = Path("/tmp/models")
        config.user_agent = "test"

        logger = MagicMock()
        thumb_manager = MagicMock()
        model_registry = MagicMock()

        params = AnalysisParameters(
            output_folder="/tmp/out",
            video_path="test.mp4",
            primary_seed_strategy="🤖 Automatic",
            compute_face_sim=True,
            pre_analysis_enabled=True,
            pre_sample_nth=1,
        )

        return config, logger, thumb_manager, model_registry, params

    @pytest.fixture
    def masker(self, mock_deps):
        config, logger, thumb_manager, model_registry, params = mock_deps

        # Mock tracker
        mock_tracker = MagicMock()
        model_registry.get_tracker.return_value = mock_tracker

        cancel_event = MagicMock()
        cancel_event.is_set.return_value = False

        masker = SubjectMasker(
            params,
            MagicMock(),  # queue
            cancel_event,
            config,
            logger=logger,
            thumbnail_manager=thumb_manager,
            model_registry=model_registry,
        )
        masker.mask_propagator = MagicMock()
        # Default return for propagator to avoid unpacking errors
        masker.mask_propagator.propagate_video.return_value = ({}, {}, {}, {})

        # Ensure thumbnail_manager returns something so _load_shot_frames doesn't skip
        masker.thumbnail_manager.get.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

        masker.seed_selector = MagicMock()
        return masker

    def test_run_propagation_happy_path(self, masker, tmp_path):
        """Test full propagation loop with video_lowres.mp4 present."""
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        (frames_dir / "video_lowres.mp4").touch()
        (frames_dir / "thumbs").mkdir()

        masker.params.output_folder = str(tmp_path / "out")

        scene = Scene(shot_id=1, start_frame=0, end_frame=10)
        scene.best_frame = 0
        scene.seed_result = {"bbox": [0, 0, 10, 10], "details": {"type": "test"}}

        masker.frame_map = {0: "frame_0.webp"}
        # Use side_effect to ensure it always returns an image
        masker.thumbnail_manager.get.side_effect = lambda x: np.zeros((100, 100, 3), dtype=np.uint8)

        # Mock propagator results
        masker.mask_propagator.propagate_video.return_value = (
            {0: np.ones((100, 100), dtype=bool)},  # masks
            {0: 0.5},  # areas
            {0: False},  # empties
            {},  # errors
        )

        with patch("core.scene_utils.subject_masker.cv2.imwrite", return_value=True):
            results = masker.run_propagation(str(frames_dir), [scene])

        assert "frame_0.png" in results
        assert results["frame_0.png"]["mask_area_pct"] == 0.5
        masker.mask_propagator.propagate_video.assert_called_once()

    def test_run_propagation_cancellation(self, masker, tmp_path):
        """Test propagation loop respects cancel_event."""
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        (frames_dir / "video_lowres.mp4").touch()

        scene1 = Scene(shot_id=1, start_frame=0, end_frame=10)
        scene1.seed_result = {"bbox": [0, 0, 10, 10]}
        scene2 = Scene(shot_id=2, start_frame=10, end_frame=20)

        masker.frame_map = {0: "f0.webp", 10: "f10.webp"}
        masker.cancel_event.is_set.side_effect = [False, True]  # Cancel after first scene

        masker.run_propagation(str(frames_dir), [scene1, scene2])

        assert masker.mask_propagator.propagate_video.call_count == 1

    def test_run_propagation_no_video_fallback(self, masker, tmp_path):
        """Test fallback warning when video_lowres.mp4 is missing."""
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()

        masker.run_propagation(str(frames_dir), [])

        # Verify warning logged (logger is second element in mock_deps)
        from unittest.mock import ANY

        masker.logger.warning.assert_any_call("video_lowres.mp4 not found, falling back to legacy mode.", extra=ANY)

    def test_load_shot_frames_missing_manager(self, masker):
        """Test _load_shot_frames early return if thumbnail_manager is None."""
        masker.thumbnail_manager = None
        masker.frame_map = {0: "f0.webp"}
        frames = masker._load_shot_frames("/tmp", Path("/tmp/thumbs"), 0, 10)
        assert frames == []

    def test_load_shot_frames_none_from_manager(self, masker):
        """Test _load_shot_frames skips frames where manager returns None."""
        masker.frame_map = {0: "f0.webp", 1: "f1.webp"}
        masker.thumbnail_manager.get.side_effect = [None, np.zeros((10, 10, 3))]
        frames = masker._load_shot_frames("/tmp", Path("/tmp/thumbs"), 0, 10)
        assert len(frames) == 1
        assert frames[0][0] == 1

    def test_select_best_frame_disabled(self, masker):
        """Test _select_best_frame_in_scene when pre-analysis is disabled."""
        masker.params.pre_analysis_enabled = False
        scene = Scene(shot_id=1, start_frame=10, end_frame=20)
        masker._select_best_frame_in_scene(scene, "/tmp")
        assert scene.best_frame == 10
        assert "reason" in scene.seed_metrics

    def test_select_best_frame_no_frames(self, masker):
        """Test _select_best_frame_in_scene when no frames are loaded."""
        scene = Scene(shot_id=1, start_frame=0, end_frame=10)
        with patch.object(masker, "_load_shot_frames", return_value=[]):
            masker._select_best_frame_in_scene(scene, "/tmp")
        assert scene.best_frame == 0
        assert scene.seed_metrics["reason"] == "no frames loaded"

    def test_select_best_frame_full_logic(self, masker):
        """Test _select_best_frame_in_scene with NIQE and Face Analyzer."""
        scene = Scene(shot_id=1, start_frame=0, end_frame=10)
        frames = [(0, np.zeros((10, 10, 3), dtype=np.uint8), (10, 10))]
        masker.niqe_metric = MagicMock()
        # Rely on torch from conftest.py
        masker.niqe_metric.return_value = torch.tensor([5.0])
        masker.niqe_metric.device = "cpu"

        masker.face_analyzer = MagicMock()
        mock_face = MagicMock()
        mock_face.det_score = 0.9
        mock_face.normed_embedding = np.ones(128)
        masker.face_analyzer.get.return_value = [mock_face]
        masker.reference_embedding = np.ones(128)

        with (
            patch.object(masker, "_load_shot_frames", return_value=frames),
            patch("torch.from_numpy", return_value=torch.zeros((1, 3, 10, 10))),
            patch("core.scene_utils.subject_masker.torch.cuda.amp.autocast"),
        ):
            masker._select_best_frame_in_scene(scene, "/tmp")

        assert scene.best_frame == 0
        assert "best_niqe" in scene.seed_metrics

    def test_get_seed_for_frame_manual(self, masker):
        """Test get_seed_for_frame with manual override."""
        seed_config = {"manual_bbox_xywh": [10, 10, 20, 20], "seed_type": "manual"}
        bbox, details = masker.get_seed_for_frame(np.zeros((10, 10, 3)), seed_config)
        assert bbox == [10, 10, 20, 20]
        assert details["type"] == "manual"

    def test_close_handles_exceptions(self, masker):
        """Test close() swallows exceptions from tracker."""
        masker.dam_tracker = MagicMock()
        masker.dam_tracker.close_session.side_effect = Exception("error")
        masker.mask_propagator = MagicMock()

        # Should not raise
        masker.close()
        masker.dam_tracker.close_session.assert_called_once()
        masker.mask_propagator.close.assert_called_once()
