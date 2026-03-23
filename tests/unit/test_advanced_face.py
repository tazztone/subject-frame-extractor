from unittest.mock import MagicMock

import numpy as np
import pytest

from core.operators.advanced_face import FaceProminenceOperator, FaceSimilarityOperator
from core.operators.base import OperatorContext


def test_face_similarity_success():
    """Test successful face similarity calculation."""
    op = FaceSimilarityOperator()

    ref_emb = np.array([1.0, 0.0, 0.0])

    mock_face = MagicMock()
    mock_face.normed_embedding = np.array([1.0, 0.0, 0.0])
    mock_face.det_score = 0.95

    ctx = OperatorContext(
        image_rgb=np.zeros((100, 100, 3)), params={"reference_embedding": ref_emb, "faces": [mock_face]}
    )

    result = op.execute(ctx)
    assert result.success
    assert result.metrics["face_sim"] == 1.0
    assert result.metrics["face_conf"] == 0.95


def test_face_similarity_missing_params():
    """Test missing parameters."""
    op = FaceSimilarityOperator()
    ctx = OperatorContext(image_rgb=np.zeros((100, 100, 3)))
    result = op.execute(ctx)
    assert result.success
    assert "Missing reference face embedding" in result.warnings


def test_face_prominence_calculation():
    """Test face prominence based on area and centrality."""
    op = FaceProminenceOperator()
    # Image 100x100
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    # Face exactly in center, size 20x20
    # Center is 50,50. Bbox [40, 40, 60, 60]
    face_bbox = [40, 40, 60, 60]

    ctx = OperatorContext(image_rgb=img, params={"face_bbox": face_bbox})

    result = op.execute(ctx)
    assert result.success
    # Area = 400. Pct = 400 / 10000 = 4%.
    # Dist from center = 0. Centrality = 1.0.
    # Score = 4 * 0.7 + 1.0 * 30.0 = 2.8 + 30.0 = 32.8
    assert result.metrics["face_prominence_score"] == pytest.approx(32.8)


def test_face_prominence_off_center():
    """Test face prominence when face is at the edge."""
    op = FaceProminenceOperator()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # Face at corner [0,0, 10,10]
    face_bbox = [0, 0, 10, 10]
    ctx = OperatorContext(image_rgb=img, params={"face_bbox": face_bbox})
    result = op.execute(ctx)
    assert result.success
    # Centrality should be lower than previous test
    assert result.metrics["face_prominence_score"] < 32.8
