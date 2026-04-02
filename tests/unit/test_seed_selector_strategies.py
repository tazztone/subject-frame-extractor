from unittest.mock import MagicMock

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from core.scene_utils.seed_selector import SeedSelector


class TestSeedSelectorStrategies:
    @pytest.fixture
    def mock_boxes(self):
        """Mock detection boxes: [x1, y1, x2, y2]"""
        return [
            {"bbox": [10, 10, 50, 50], "conf": 0.8, "type": "person"},  # Moderate
            {"bbox": [100, 100, 120, 120], "conf": 0.95, "type": "person"},  # Small, high conf
            {"bbox": [400, 400, 450, 480], "conf": 0.7, "type": "person"},  # Tall
        ]

    @pytest.fixture
    def mock_frame(self):
        return np.zeros((500, 500, 3), dtype=np.uint8)

    def _get_selector(self, mock_config, mock_logger, strategy="Largest Subject"):
        params_dict = {
            "primary_seed_strategy": strategy,
            "seed_strategy": strategy,  # SeedSelector uses getattr(params, "seed_strategy", ...) or params.get("seed_strategy", ...)
            "compute_face_sim": False,
            "text_prompt": "",
        }

        return SeedSelector(params=params_dict, config=mock_config, logger=mock_logger, device="cpu")

    @pytest.mark.parametrize(
        "strategy, expected_fragment",
        [
            ("Largest Subject", "largest_subject"),
            ("Center-most Subject", "center-most_subject"),
            ("Highest Confidence", "highest_confidence"),
            ("Tallest Subject", "tallest_subject"),
            ("Area x Confidence", "area_x_confidence"),
            ("Rule-of-Thirds", "rule-of-thirds"),
            ("Edge-avoiding", "edge-avoiding"),
            ("Balanced", "balanced"),
        ],
    )
    def test_all_strategies_dispatch(
        self, strategy, expected_fragment, mock_frame, mock_boxes, mock_config, mock_logger
    ):
        selector = self._get_selector(mock_config, mock_logger, strategy)
        selector.tracker = MagicMock()
        selector.tracker.detect_objects.return_value = mock_boxes

        result_box, result_meta = selector.select_seed(mock_frame)

        assert result_box is not None
        assert expected_fragment in result_meta["type"]

    def test_seed_selector_best_face_strategy(self, mock_frame, mock_boxes, mock_config, mock_logger):
        """Test 'Best Face' strategy specifically."""
        strategy = "Best Face"
        selector = self._get_selector(mock_config, mock_logger, strategy)
        selector.tracker = MagicMock()
        selector.tracker.detect_objects.return_value = mock_boxes

        # Mock face analyzer
        mock_face = MagicMock()
        mock_face.bbox = np.array([15, 15, 30, 30])
        mock_face.det_score = 0.99
        selector.face_analyzer = MagicMock()
        selector.face_analyzer.get.return_value = [mock_face]

        result_box, result_meta = selector.select_seed(mock_frame)

        assert result_box is not None
        assert "best_face" in result_meta["type"]

    def test_seed_selector_fallback_to_face(self, mock_frame, mock_config, mock_logger):
        """Test fallback to face when no subjects are found."""
        selector = self._get_selector(mock_config, mock_logger)
        selector.tracker = MagicMock()
        selector.tracker.detect_objects.return_value = []  # No persons

        # Mock face analyzer
        mock_face = MagicMock()
        mock_face.bbox = np.array([50, 50, 100, 100])
        mock_face.det_score = 0.9
        selector.face_analyzer = MagicMock()
        selector.face_analyzer.get.return_value = [mock_face]

        result_box, result_meta = selector.select_seed(mock_frame)

        assert result_box is not None
        assert "face_fallback" in result_meta["type"]

    def test_seed_selector_final_fallback(self, mock_frame, mock_config, mock_logger):
        """Test final fallback when neither subjects nor faces are found."""
        selector = self._get_selector(mock_config, mock_logger)
        selector.tracker = MagicMock()
        selector.tracker.detect_objects.return_value = []
        selector.face_analyzer = MagicMock()
        selector.face_analyzer.get.return_value = []  # No faces

        result_box, result_meta = selector.select_seed(mock_frame)

        assert result_box is not None
        assert "no_subjects_fallback" in result_meta["type"]

    @given(
        x1=st.floats(-1000, 1000),
        y1=st.floats(-1000, 1000),
        x2=st.floats(-1000, 1000),
        y2=st.floats(-1000, 1000),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50)
    def test_xyxy_to_xywh_robustness(self, x1, y1, x2, y2, mock_config, mock_logger):
        """Ensure coordinate conversion never crashes and handles inverted boxes."""
        selector = self._get_selector(mock_config, mock_logger)
        res = selector._xyxy_to_xywh([x1, y1, x2, y2])
        assert len(res) == 4

    @given(
        box1=st.lists(st.floats(0, 1000), min_size=4, max_size=4),
        box2=st.lists(st.floats(0, 1000), min_size=4, max_size=4),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture], max_examples=50)
    def test_iou_always_in_range(self, box1, box2, mock_config, mock_logger):
        """IoU must always be in [0, 1] for non-negative coordinates."""
        selector = self._get_selector(mock_config, mock_logger)
        # Ensure x2 >= x1 and y2 >= y1 for a valid box representation
        b1 = [min(box1[0], box1[2]), min(box1[1], box1[3]), max(box1[0], box1[2]), max(box1[1], box1[3])]
        b2 = [min(box2[0], box2[2]), min(box2[1], box2[3]), max(box2[0], box2[2]), max(box2[1], box2[3])]

        iou = selector._calculate_iou(b1, b2)
        assert 0.0 <= iou <= 1.000001  # Account for float precision
