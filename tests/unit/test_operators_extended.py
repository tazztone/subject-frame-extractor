from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.operators import OperatorContext
from core.operators.dedup import apply_deduplication_filter
from core.operators.face_metrics import EyesOpenOperator, FacePoseOperator


def test_eyes_open_operator_basic(mock_config, sample_image, mock_logger):
    """Test EyesOpenOperator with mock MediaPipe blendshapes."""
    ctx = OperatorContext(
        image_rgb=sample_image,
        config=mock_config,
        logger=mock_logger,
        params={"face_blendshapes": {"eyeBlinkLeft": 0.1, "eyeBlinkRight": 0.2}},
    )
    op = EyesOpenOperator()
    result = op.execute(ctx)

    # 1.0 - avg(0.1, 0.2) = 0.85
    assert "eyes_open" in result.metrics
    assert pytest.approx(result.metrics["eyes_open"], 0.1) == 0.85


def test_face_pose_operator_basic(mock_config, sample_image, mock_logger):
    """Test FacePoseOperator with mock transformation matrix."""
    # Identity matrix means looking straight ahead (0, 0, 0)
    identity_matrix = np.eye(4)
    ctx = OperatorContext(
        image_rgb=sample_image, config=mock_config, logger=mock_logger, params={"face_matrix": identity_matrix}
    )
    op = FacePoseOperator()
    result = op.execute(ctx)

    assert "yaw" in result.metrics
    assert "pitch" in result.metrics
    assert "roll" in result.metrics
    assert result.metrics["yaw"] == 0.0


def test_dedup_phash_basic(mock_config, mock_logger):
    """Test pHash deduplication filter logic."""
    frames_data = [
        {"filename": "f1.png", "phash": "0000000000000000"},
        {"filename": "f2.png", "phash": "0000000000000001"},  # 1 bit diff
        {"filename": "f3.png", "phash": "ffffffffffffffff"},  # Very diff
    ]
    filters = {"enable_dedup": True, "dedup_method": "pHash", "dedup_thresh": 5}

    with patch("imagehash.hex_to_hash") as mock_hex:
        # Mock the hash objects
        h1 = MagicMock()
        h1.hash = np.zeros((8, 8), dtype=bool)
        h2 = MagicMock()
        h2.hash = np.zeros((8, 8), dtype=bool)
        h2.hash[0, 0] = True  # 1 bit diff
        h3 = MagicMock()
        h3.hash = np.ones((8, 8), dtype=bool)

        # imagehash distance is __sub__
        h1.__sub__.side_effect = lambda other: 1 if other == h2 else 64

        mock_hex.side_effect = [h1, h2, h3]

        # The internal kept_hash_matrix size depends on the first hash found
        mask, reasons = apply_deduplication_filter(frames_data, filters, MagicMock(), mock_config, "/tmp")
        # f1 and f2 are duplicates (dist 1 < thresh 5). f1 is kept, f2 dropped.
        assert mask[0] == True
        assert mask[1] == False
        assert mask[2] == True
        assert "duplicate" in reasons["f2.png"]
