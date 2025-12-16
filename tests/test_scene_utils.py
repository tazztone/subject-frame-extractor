import pytest
from unittest.mock import MagicMock, patch, ANY, mock_open
import sys
from pathlib import Path
from queue import Queue
import threading
import json
import numpy as np
import torch

from core.config import Config
from core.models import AnalysisParameters, Scene
from core.scene_utils_pkg import SeedSelector, MaskPropagator, SubjectMasker, run_scene_detection

@pytest.fixture
def mock_config(tmp_path):
    config = MagicMock() # Removed spec=Config to avoid attribute issues
    config.downloads_dir = tmp_path / "downloads"
    config.models_dir = tmp_path / "models"
    config.retry_max_attempts = 1
    config.retry_backoff_seconds = (0.1,)
    config.seeding_yolo_iou_threshold = 0.5
    config.seeding_face_contain_score = 10
    config.seeding_confidence_score_multiplier = 1
    config.seeding_iou_bonus = 5
    config.seeding_balanced_score_weights = {'area': 1, 'confidence': 1, 'edge': 1}
    config.seeding_face_to_body_expansion_factors = [1.5, 3.0, 1.0] # w, h, top
    config.seeding_final_fallback_box = [0.25, 0.25, 0.75, 0.75]
    config.visualization_bbox_color = (0, 255, 0)
    config.visualization_bbox_thickness = 2
    return config

@pytest.fixture
def mock_logger():
    return MagicMock()

@pytest.fixture
def mock_params():
    p = AnalysisParameters(
        source_path="test.mp4",
        output_folder="/tmp/out",
        tracker_model_name="sam3"
    )
    p.seed_strategy = "Largest Person"
    return p

class TestSeedSelector:
    def test_select_seed_largest_person(self, mock_config, mock_logger, mock_params):
        tracker = MagicMock()
        # Mock detections: [x1, y1, x2, y2], conf, type
        detections = [
            {'bbox': [0, 0, 10, 10], 'conf': 0.9, 'type': 'person'}, # Area 100
            {'bbox': [0, 0, 20, 20], 'conf': 0.8, 'type': 'person'}, # Area 400 (Winner)
        ]
        tracker.detect_objects.return_value = detections

        selector = SeedSelector(mock_params, mock_config, None, None, tracker, mock_logger)

        frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox, details = selector.select_seed(frame_rgb)

        assert bbox == [0, 0, 20, 20] # xywh: 0, 0, 20, 20
        assert details['conf'] == 0.8
        assert 'person_largest_person' == details['type']

    def test_select_seed_text_prompt(self, mock_config, mock_logger, mock_params):
        mock_params.primary_seed_strategy = "📝 By Text"
        mock_params.text_prompt = "cat"

        tracker = MagicMock()
        detections = [
            {'bbox': [10, 10, 30, 30], 'conf': 0.95, 'type': 'cat'}
        ]

        def side_effect(frame, prompt):
            if prompt == "person": return []
            return detections
        tracker.detect_objects.side_effect = side_effect

        selector = SeedSelector(mock_params, mock_config, None, None, tracker, mock_logger)
        frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)

        bbox, details = selector.select_seed(frame_rgb)
        assert bbox == [10, 10, 20, 20]
        assert details['type'] == 'cat'

class TestMaskPropagator:
    @patch('core.scene_utils_pkg.mask_propagator.postprocess_mask', side_effect=lambda x, **k: x)
    def test_propagate_success(self, mock_post, mock_config, mock_logger, mock_params):
        tracker = MagicMock()
        # Mock initialize
        tracker.initialize.return_value = {'pred_mask': np.ones((100, 100), dtype=float)}
        # Mock propagate_from
        tracker.propagate_from.return_value = [] # No propagation for simplicity or mock it

        propagator = MaskPropagator(mock_params, tracker, threading.Event(), Queue(), mock_config, mock_logger)

        frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        masks, areas, empties, errors = propagator.propagate(frames, 0, [0, 0, 10, 10])

        assert len(masks) == 1
        assert masks[0] is not None
        assert areas[0] > 0
        assert not empties[0]
        assert errors[0] is None

class TestSubjectMasker:
    @patch('core.scene_utils_pkg.subject_masker.create_frame_map', return_value={0: 'frame_0.png'})
    def test_run_propagation(self, mock_create_map, mock_config, mock_logger, mock_params, tmp_path):
        mock_model_registry = MagicMock()
        mock_tracker = MagicMock()
        mock_model_registry.get_tracker.return_value = mock_tracker

        # Mock propagator
        with patch('core.scene_utils_pkg.subject_masker.MaskPropagator') as MockPropagator:
            instance = MockPropagator.return_value
            # return masks, areas, empties, errors
            instance.propagate.return_value = ([np.ones((10, 10), dtype=np.uint8)], [100.0], [False], [None])

            masker = SubjectMasker(mock_params, Queue(), threading.Event(), mock_config, logger=mock_logger, model_registry=mock_model_registry)
            masker.frame_map = {0: 'frame_0.png'}

            # Setup scene
            scene = Scene(shot_id=1, start_frame=0, end_frame=1, best_frame=0, seed_result={'bbox': [0,0,10,10], 'details': {}})

            # Setup disk mocks
            with patch('core.scene_utils_pkg.subject_masker.SubjectMasker._load_shot_frames') as mock_load:
                mock_load.return_value = [(0, np.zeros((10,10,3), dtype=np.uint8), (10,10))]

                frames_dir = tmp_path / "frames"
                frames_dir.mkdir()

                result = masker.run_propagation(str(frames_dir), [scene])

                assert result
                assert 'frame_0.png' in result
                assert result['frame_0.png']['mask_path'] is not None
