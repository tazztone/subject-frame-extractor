from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.enums import SeedStrategy
from core.events import PreAnalysisEvent
from core.models import AnalysisParameters
from core.scene_utils.seed_selector import SeedSelector


class TestStrategyMapping:
    @pytest.fixture
    def mock_config(self, mock_config_simple):
        return mock_config_simple

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    @pytest.fixture
    def selector(self, mock_config, mock_logger, mock_params):
        face_analyzer = MagicMock()
        tracker = MagicMock()
        return SeedSelector(mock_params, mock_config, face_analyzer, None, tracker, mock_logger)

    def test_pre_analysis_event_handles_emoji_string(self, mock_logger, mock_config):
        """
        FAILING TEST: Currently, PreAnalysisEvent does not automatically strip emojis
        from strategy strings. It relies on the UI's LEGACY_STRATEGY_MAP.
        This test will fail if the UI sends "👤 Source Face Reference"
        unless we add a validator to the model.
        """
        raw_args = {
            "output_folder": "/tmp/out",
            "video_path": "video.mp4",
            "primary_seed_strategy": "👤 Source Face Reference",
        }
        # This will currently fail to match 'Source Face Reference'
        # unless normalized.
        event = PreAnalysisEvent.model_validate(raw_args)
        assert event.primary_seed_strategy == SeedStrategy.FACE_REFERENCE.value

    def test_seed_selector_routes_face_correctly(self, selector):
        """
        GREEN TEST: Now that AnalysisParameters has a validator, or we
        simulate it by ensuring SeedSelector works with clean strings.
        """
        # Create an AnalysisParameters with the emoji string (it should strip it)
        params = AnalysisParameters(
            primary_seed_strategy="👤 Source Face Reference",
            output_folder="/tmp/out",
            video_path="video.mp4",
            compute_face_sim=True,
        )
        assert params.primary_seed_strategy == "Source Face Reference"

        selector.params = params
        selector.reference_embedding = np.array([1, 2, 3])
        frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch.object(selector, "_identity_first_seed") as mock_identity:
            mock_identity.return_value = ([0, 0, 10, 10], {"type": "face"})
            selector.select_seed(frame_rgb)
            assert mock_identity.called

    def test_automatic_fallback_prefers_best_face(self, selector):
        """
        GREEN TEST: In Automatic fallback, face detection now picks
        the face most similar to the reference.
        """
        selector.config.seeding_face_to_body_expansion_factors = [1.0, 1.0, 0.0]  # No expansion for easy test

        face1 = MagicMock()
        face1.det_score = 0.9
        face1.normed_embedding = np.array([1, 0, 0])
        face1.bbox = np.array([10, 10, 20, 20])

        face2 = MagicMock()
        face2.det_score = 0.7
        face2.normed_embedding = np.array([0, 1, 0])
        face2.bbox = np.array([30, 30, 40, 40])

        selector.face_analyzer.get.return_value = [face1, face2]
        selector.reference_embedding = np.array([0, 1, 0])
        selector.params.primary_seed_strategy = "Automatic Detection"

        frame_rgb = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch.object(selector, "_get_person_boxes", return_value=[]):
            box, details = selector._choose_person_by_strategy(frame_rgb, selector.params)
            # face2 bbox [30, 30, 40, 40] xyxy -> xywh [30, 30, 10, 10]
            assert box == [30, 30, 10, 10]
