import pytest
import numpy as np
import math
from core.operators import OperatorContext, OperatorResult
from core.operators.face_metrics import EyesOpenOperator, FacePoseOperator

class TestEyesOpenOperator:
    @pytest.fixture
    def operator(self):
        return EyesOpenOperator()

    @pytest.fixture
    def sample_image(self):
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def test_config(self, operator):
        assert operator.config.name == "eyes_open"
        assert operator.config.requires_face is True

    def test_missing_data_returns_warning(self, operator, sample_image):
        ctx = OperatorContext(image_rgb=sample_image, params={})
        result = operator.execute(ctx)
        assert result.success is True
        assert len(result.warnings) > 0
        assert "eyes_open_score" not in result.metrics

    def test_eyes_open(self, operator, sample_image):
        """Blink 0.0 -> Score 100."""
        params = {
            "face_blendshapes": {
                "eyeBlinkLeft": 0.0,
                "eyeBlinkRight": 0.0
            }
        }
        ctx = OperatorContext(image_rgb=sample_image, params=params)
        result = operator.execute(ctx)
        
        assert result.metrics["eyes_open_score"] == 100.0
        assert result.metrics["blink_prob"] == 0.0

    def test_eyes_closed(self, operator, sample_image):
        """Blink 1.0 -> Score 0."""
        params = {
            "face_blendshapes": {
                "eyeBlinkLeft": 1.0, 
                "eyeBlinkRight": 1.0
            }
        }
        ctx = OperatorContext(image_rgb=sample_image, params=params)
        result = operator.execute(ctx)
        
        assert result.metrics["eyes_open_score"] == 0.0
        assert result.metrics["blink_prob"] == 1.0

    def test_mixed_blink(self, operator, sample_image):
        """Max(left, right) used."""
        params = {
            "face_blendshapes": {
                "eyeBlinkLeft": 0.2, 
                "eyeBlinkRight": 0.8
            }
        }
        ctx = OperatorContext(image_rgb=sample_image, params=params)
        result = operator.execute(ctx)
        
        # Max is 0.8. Score = 1.0 - 0.8 = 0.2 -> 20.0
        assert result.metrics["eyes_open_score"] == pytest.approx(20.0)
        assert result.metrics["blink_prob"] == 0.8


class TestFacePoseOperator:
    @pytest.fixture
    def operator(self):
        return FacePoseOperator()
        
    @pytest.fixture
    def sample_image(self):
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def test_config(self, operator):
        assert operator.config.name == "face_pose"

    def test_missing_matrix(self, operator, sample_image):
        ctx = OperatorContext(image_rgb=sample_image, params={})
        result = operator.execute(ctx)
        assert "Missing face_matrix" in result.warnings[0]

    def test_identity_matrix(self, operator, sample_image):
        """Identity matrix -> all 0 angles."""
        matrix = np.eye(4)
        params = {"face_matrix": matrix}
        ctx = OperatorContext(image_rgb=sample_image, params=params)
        result = operator.execute(ctx)
        
        assert result.metrics["yaw"] == 0.0
        assert result.metrics["pitch"] == 0.0
        assert result.metrics["roll"] == 0.0

    def test_yaw_rotation(self, operator, sample_image):
        """Rotate around Y axis (approx)."""
        # Rotation 90 degrees around Y:
        # matrix[0,0] = cos(90) = 0
        # matrix[1,0] = 0
        # Wait, standard rotation matrix logic:
        # R_y(theta) = [cos 0 sin; 0 1 0; -sin 0 cos]
        # models.py: yaw = atan2(matrix[1,0], matrix[0,0]) ?
        # Wait, check models.py logic:
        # yaw = degrees(atan2(matrix[1, 0], matrix[0, 0]))
        # If identity: [0,0]=1, [1,0]=0. atan2(0, 1) = 0.
        # If 90 deg: [0,0]=0, [1,0]=1 (depends on axis convention).
        
        # Let's trust the math implementation is consistent with models.py
        # Test distinct values
        matrix = np.eye(4)
        matrix[0,0] = 0.5
        matrix[1,0] = 0.5
        # atan2(0.5, 0.5) = 45 degrees
        
        params = {"face_matrix": matrix}
        ctx = OperatorContext(image_rgb=sample_image, params=params)
        result = operator.execute(ctx)
        
        assert result.metrics["yaw"] == pytest.approx(45.0)
