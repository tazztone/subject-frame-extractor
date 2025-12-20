"""
Tests for scene utilities - SeedSelector, MaskPropagator, SubjectMasker.

Uses fixtures from conftest.py for mock setup.
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from queue import Queue
import threading
import numpy as np

from core.config import Config
from core.models import AnalysisParameters, Scene
from core.scene_utils import SeedSelector, MaskPropagator, SubjectMasker, run_scene_detection


class TestSeedSelector:
    def test_select_seed_largest_person(self, mock_config_simple, mock_logger, mock_params):
        tracker = MagicMock()
        # Mock detections: [x1, y1, x2, y2], conf, type
        detections = [
            {'bbox': [0, 0, 10, 10], 'conf': 0.9, 'type': 'person'},  # Area 100
            {'bbox': [0, 0, 20, 20], 'conf': 0.8, 'type': 'person'},  # Area 400 (Winner)
        ]
        tracker.detect_objects.return_value = detections

        selector = SeedSelector(mock_params, mock_config_simple, None, None, tracker, mock_logger)

        frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox, details = selector.select_seed(frame_rgb)

        assert bbox == [0, 0, 20, 20]  # xywh: 0, 0, 20, 20
        assert details['conf'] == 0.8
        assert 'person' in details['type']  # Type should contain 'person'

    def test_select_seed_text_prompt(self, mock_config_simple, mock_logger, mock_params):
        mock_params.primary_seed_strategy = "📝 By Text"
        mock_params.text_prompt = "cat"

        tracker = MagicMock()
        detections = [
            {'bbox': [10, 10, 30, 30], 'conf': 0.95, 'type': 'cat'}
        ]

        def side_effect(frame, prompt):
            if prompt == "person":
                return []
            return detections
        tracker.detect_objects.side_effect = side_effect

        selector = SeedSelector(mock_params, mock_config_simple, None, None, tracker, mock_logger)
        frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)

        bbox, details = selector.select_seed(frame_rgb)
        assert bbox == [10, 10, 20, 20]
        assert details['type'] == 'cat'


class TestMaskPropagator:
    @patch('core.scene_utils.mask_propagator.postprocess_mask', side_effect=lambda x, **k: x)
    def test_propagate_success(self, mock_post, mock_config_simple, mock_logger, mock_params):
        tracker = MagicMock()
        # Mock new SAM3 API
        tracker.init_video.return_value = None
        tracker.add_bbox_prompt.return_value = np.ones((100, 100), dtype=bool)
        tracker.propagate.return_value = [] # Yields nothing for simplicity

        propagator = MaskPropagator(mock_params, tracker, threading.Event(), Queue(), mock_config_simple, mock_logger)

        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        masks, areas, empties, errors = propagator.propagate(frames, 0, [0, 0, 10, 10])

        assert len(masks) == 1
        assert masks[0] is not None
        assert areas[0] > 0
        assert not empties[0]
        assert errors[0] is None


class TestSubjectMasker:
    @patch('core.scene_utils.subject_masker.create_frame_map', return_value={0: 'frame_0.png'})
    def test_run_propagation(self, mock_create_map, mock_config_simple, mock_logger, mock_params, tmp_path):
        mock_model_registry = MagicMock()
        mock_tracker = MagicMock()
        mock_model_registry.get_tracker.return_value = mock_tracker

        # Mock propagator
        with patch('core.scene_utils.subject_masker.MaskPropagator') as MockPropagator:
            instance = MockPropagator.return_value
            # return masks, areas, empties, errors
            instance.propagate.return_value = ([np.ones((10, 10), dtype=np.uint8)], [100.0], [False], [None])

            masker = SubjectMasker(mock_params, Queue(), threading.Event(), mock_config_simple, logger=mock_logger, model_registry=mock_model_registry)
            masker.frame_map = {0: 'frame_0.png'}

            # Setup scene
            scene = Scene(shot_id=1, start_frame=0, end_frame=1, best_frame=0, seed_result={'bbox': [0, 0, 10, 10], 'details': {}})

            # Setup disk mocks
            with patch('core.scene_utils.subject_masker.SubjectMasker._load_shot_frames') as mock_load:
                mock_load.return_value = [(0, np.zeros((10, 10, 3), dtype=np.uint8), (10, 10))]

                frames_dir = tmp_path / "frames"
                frames_dir.mkdir()

                result = masker.run_propagation(str(frames_dir), [scene])

                assert result
                assert 'frame_0.png' in result
                assert result['frame_0.png']['mask_path'] is not None
