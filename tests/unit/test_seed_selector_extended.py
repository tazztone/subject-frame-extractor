"""
Extended tests for SeedSelector to improve coverage.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.scene_utils.seed_selector import SeedSelector


class TestSeedSelectorExtended:
    @pytest.fixture
    def selector(self, mock_config_simple, mock_logger, mock_params):
        tracker = MagicMock()
        face_analyzer = MagicMock()
        return SeedSelector(mock_params, mock_config_simple, face_analyzer, None, tracker, mock_logger)

    def test_identity_first_seed_no_face(self, selector):
        with patch.object(selector, "_find_target_face", return_value=(None, {"error": "no_face"})):
            frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
            box, details = selector._identity_first_seed(frame_rgb, selector.params)
            assert box is None
            assert details["type"] == "no_subject_found"

    def test_object_first_seed_success(self, selector):
        selector.tracker.detect_objects.side_effect = [
            [{"bbox": [10, 10, 20, 20], "conf": 0.8, "type": "text"}],  # text boxes
            [{"bbox": [5, 5, 25, 25], "conf": 0.9, "type": "person"}],  # person boxes
        ]
        selector.params.text_prompt = "person in red"
        selector.config.seeding_iou_threshold = 0.1

        frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        box, details = selector._object_first_seed(frame_rgb, selector.params)

        assert box is not None
        assert details["type"] == "sam3_intersect"
        assert details["iou"] > 0.1

    def test_find_target_face_match(self, selector):
        mock_face = MagicMock()
        mock_face.normed_embedding = np.ones(128)
        mock_face.bbox = np.array([10, 10, 30, 30])
        selector.face_analyzer.get.return_value = [mock_face]
        selector.reference_embedding = np.ones(128)
        selector.config.seeding_face_similarity_threshold = 0.5

        frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        face, details = selector._find_target_face(frame_rgb)

        assert face is not None
        assert details["seed_face_sim"] == 128.0  # dot product of ones

    def test_expand_face_to_body(self, selector):
        selector.config.seeding_face_to_body_expansion_factors = [3.0, 6.0, 1.0]  # w, h, top_y_offset
        face_bbox = [10, 10, 20, 20]  # w=10, h=10, cy=15, cx=15
        img_shape = (100, 100, 3)

        expanded = selector._expand_face_to_body(face_bbox, img_shape)
        # cx=15, w=10*3=30 -> x1 = 15-15=0
        # y1=10, h=10*1=10 offset -> new_y1 = 10-10=0
        # h=10*6=60
        assert expanded == [0, 0, 30, 60]

    def test_xyxy_to_xywh_padding(self, selector):
        box = [10, 10, 20, 20]  # w=10, h=10
        img_shape = (100, 100, 3)
        # 5% padding = 0.5px
        xywh = selector._xyxy_to_xywh(box, img_shape)
        # x1 = 10 - 0.5 = 9.5 -> 9
        # y1 = 10 - 0.5 = 9.5 -> 9
        # w = 20.5 - 9.5 = 11
        assert xywh == [9, 9, 11, 11]

    def test_choose_person_by_strategy_fallback(self, selector):
        selector.tracker.detect_objects.return_value = []
        frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        box, details = selector._choose_person_by_strategy(frame_rgb, selector.params)
        assert details["type"] == "no_people_fallback"

    def test_calculate_iou(self, selector):
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        iou = selector._calculate_iou(box1, box2)
        # intersection = 5x5 = 25
        # union = 100 + 100 - 25 = 175
        assert iou == pytest.approx(25 / 175)

    def test_box_contains(self, selector):
        assert selector._box_contains([0, 0, 10, 10], [2, 2, 8, 8]) is True
        assert selector._box_contains([0, 0, 10, 10], [2, 2, 12, 8]) is False
