from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.models import AnalysisParameters
from core.scene_utils.seed_selector import SeedSelector


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_params():
    return AnalysisParameters(source_path="v.mp4")


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.device = "cuda"
    config.seeding_face_similarity_threshold = 0.5
    return config


@pytest.fixture
def selector(mock_params, mock_config, mock_logger):
    face_analyzer = MagicMock()
    tracker = MagicMock()
    # SeedSelector(params, config, face_analyzer, ref_emb, tracker, logger)
    return SeedSelector(mock_params, mock_config, face_analyzer, None, tracker, mock_logger)


def test_get_mask_for_bbox_error_paths(selector):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    bbox = [0, 0, 5, 5]

    # 1. No tracker
    selector.tracker = None
    assert selector._get_mask_for_bbox(frame, bbox) is None

    # 2. GPU OOM
    selector.tracker = MagicMock()
    # Force the tracker to raise "out of memory"
    selector.tracker.init_video.side_effect = RuntimeError("out of memory")

    with patch("core.scene_utils.seed_selector.is_cuda_available", return_value=True):
        with patch("core.scene_utils.seed_selector.empty_cache") as mock_empty:
            selector.device = "cuda"  # Explicitly set for test
            assert selector._get_mask_for_bbox(frame, bbox) is None
            mock_empty.assert_called_once()

    # 3. Generic Exception
    selector.tracker.init_video.side_effect = Exception("Generic Fail")
    assert selector._get_mask_for_bbox(frame, bbox) is None
