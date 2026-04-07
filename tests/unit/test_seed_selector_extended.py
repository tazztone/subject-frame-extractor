from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.enums import SeedStrategy
from core.models import AnalysisParameters
from core.scene_utils.seed_selector import SeedSelector


class TestSeedSelectorExtended:
    @pytest.fixture
    def selector(self, mock_config_simple, mock_logger):
        params = AnalysisParameters(
            output_folder="/tmp",
            video_path="test.mp4",
            primary_seed_strategy="🤖 Automatic",
        )
        tracker = MagicMock()
        face_analyzer = MagicMock()
        return SeedSelector(params, mock_config_simple, face_analyzer, None, tracker, mock_logger)

    @pytest.mark.parametrize("strategy", [s.value for s in SeedStrategy])
    def test_select_seed_all_strategies(self, selector, strategy):
        """Parametrized test for all SeedStrategy enum values."""
        selector.params.primary_seed_strategy = strategy
        frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)

        # Mock strategy methods to avoid deep testing here, just routing
        with (
            patch.object(selector, "_identity_first_seed", return_value=(None, {})),
            patch.object(selector, "_object_first_seed", return_value=(None, {})),
            patch.object(selector, "_face_with_text_fallback_seed", return_value=(None, {})),
            patch.object(selector, "_choose_subject_by_strategy", return_value=(None, {})),
        ):
            selector.select_seed(frame_rgb)

            if strategy == SeedStrategy.FACE_REFERENCE.value:
                # Should fallback because face_analyzer and reference_embedding not set together in fixture
                # unless we set them now
                selector.face_analyzer = MagicMock()
                selector.reference_embedding = np.ones(128)
                selector.params.compute_face_sim = True
                selector.select_seed(frame_rgb)
                selector._identity_first_seed.assert_called()

    def test_identity_first_seed_no_ref(self, selector):
        """IDENTITY_FIRST with reference_embedding=None falls back to object path."""
        selector.reference_embedding = None
        selector.params.primary_seed_strategy = SeedStrategy.FACE_REFERENCE.value

        with patch.object(selector, "_object_first_seed", return_value=([0, 0, 1, 1], {})) as mock_obj:
            frame_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
            selector.select_seed(frame_rgb)
            mock_obj.assert_called_once()

    def test_identity_first_with_match(self, selector):
        """IDENTITY_FIRST with match and person detection."""
        selector.reference_embedding = np.ones(128)
        mock_face = MagicMock()
        mock_face.normed_embedding = np.ones(128)
        mock_face.bbox = np.array([10, 10, 20, 20])
        selector.face_analyzer.get.return_value = [mock_face]

        # Mock person boxes
        selector.tracker.detect_objects.return_value = [{"bbox": [5, 5, 25, 25], "conf": 0.9, "type": "person"}]

        frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        box, details = selector._identity_first_seed(frame_rgb, selector.params)

        assert box is not None
        assert details["type"] == "evidence_based_selection"

    def test_identity_first_no_match_fallback_expansion(self, selector):
        """IDENTITY_FIRST finds face but no person box -> expansion."""
        selector.reference_embedding = np.ones(128)
        mock_face = MagicMock()
        mock_face.normed_embedding = np.ones(128)
        mock_face.bbox = np.array([10, 10, 20, 20])
        selector.face_analyzer.get.return_value = [mock_face]

        # No person boxes, no text boxes
        selector.tracker.detect_objects.return_value = []

        frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        box, details = selector._identity_first_seed(frame_rgb, selector.params)

        assert box is not None
        assert details["type"] == "expanded_box_from_face"

    def test_object_first_seed_no_intersect(self, selector):
        """OBJECT_FIRST with text prompt but no person validation."""
        selector.tracker.detect_objects.side_effect = [
            [{"bbox": [10, 10, 20, 20], "conf": 0.8, "type": "text"}],  # text boxes
            [],  # no person boxes
        ]
        selector.params.text_prompt = "cat"

        frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        box, details = selector._object_first_seed(frame_rgb, selector.params)

        assert box == [9, 9, 11, 11]  # padded xywh
        assert "all_boxes_count" in details

    def test_expand_face_to_body_edge_cases(self, selector):
        """Test expansion logic at image boundaries."""
        img_shape = (100, 100, 3)
        selector.config.seeding_face_to_body_expansion_factors = [2.0, 4.0, 1.0]

        # Face at top-left corner
        face_bbox = [0, 0, 10, 10]
        expanded = selector._expand_face_to_body(face_bbox, img_shape)
        assert expanded[0] == 0  # Clamped x1
        assert expanded[1] == 0  # Clamped y1

        # Face at bottom-right corner
        face_bbox = [90, 90, 100, 100]
        expanded = selector._expand_face_to_body(face_bbox, img_shape)
        assert expanded[0] + expanded[2] <= 100
        assert expanded[1] + expanded[3] <= 100

    def test_choose_subject_by_strategy_all_variants(self, selector):
        """Test different selection strategies in _choose_subject_by_strategy."""
        strategies = [
            "Largest Object",
            "Center-most Object",
            "Highest Confidence",
            "Tallest Object",
            "Area x Confidence",
            "Rule-of-Thirds",
            "Edge-avoiding",
            "Balanced",
            "Best Face",
        ]

        boxes = [
            {"bbox": [0, 0, 10, 10], "conf": 0.5, "type": "person"},
            {"bbox": [40, 40, 60, 60], "conf": 0.9, "type": "person"},
        ]
        selector.tracker.detect_objects.return_value = boxes
        frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)

        for strat in strategies:
            selector.params.seed_strategy = strat
            box, details = selector._choose_subject_by_strategy(frame_rgb, selector.params)
            assert box is not None

    def test_choose_subject_by_strategy_face_fallback(self, selector):
        """Fallback to face detection when no subjects are detected."""
        selector.tracker.detect_objects.return_value = []
        mock_face = MagicMock()
        mock_face.bbox = np.array([10, 10, 20, 20])
        mock_face.det_score = 0.95
        selector.face_analyzer.get.return_value = [mock_face]

        frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        box, details = selector._choose_subject_by_strategy(frame_rgb, selector.params)

        assert details["type"] == "face_fallback_expanded"
        assert details["face_conf"] == 0.95

    def test_get_mask_for_bbox_torch_oom(self, selector):
        """Test OOM handling in _get_mask_for_bbox."""

        with (
            patch("core.scene_utils.seed_selector.is_cuda_available", return_value=True),
            patch("core.scene_utils.seed_selector.empty_cache") as mock_empty,
        ):
            selector.tracker.init_video.side_effect = RuntimeError("out of memory")

            mask = selector._get_mask_for_bbox(np.zeros((10, 10, 3), dtype=np.uint8), [0, 0, 5, 5])
            assert mask is None
            mock_empty.assert_called_once()
