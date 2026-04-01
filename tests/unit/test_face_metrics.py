from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.operators import OperatorContext
from core.operators.face_metrics import EyesOpenOperator, FacePoseOperator


class TestFaceMetrics:
    @pytest.fixture
    def mock_context(self):
        ctx = MagicMock(spec=OperatorContext)
        ctx.shared_data = {}
        ctx.params = {}
        ctx.image_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        ctx.config = MagicMock()
        ctx.logger = MagicMock()
        return ctx

    def test_eyes_open_operator(self, mock_context):
        op = EyesOpenOperator()

        # Inject mock blendshapes into params for _get_face_data
        mock_context.params["face_blendshapes"] = {"eyeBlinkLeft": 0.1, "eyeBlinkRight": 0.2}

        result = op.execute(mock_context)
        assert "eyes_open" in result.metrics
        assert result.metrics["eyes_open"] == 0.8  # 1.0 - max(0.1, 0.2)
        assert result.metrics["eyes_open_score"] == 80.0

    def test_face_pose_operator(self, mock_context):
        op = FacePoseOperator()

        # Identity matrix (looking straight)
        matrix = np.eye(4)
        mock_context.params["face_matrix"] = matrix

        result = op.execute(mock_context)
        assert "yaw" in result.metrics
        assert "pitch" in result.metrics
        assert result.metrics["yaw"] == 0.0
        assert result.metrics["pitch"] == 0.0

    def test_get_face_data_real_path(self, mock_context):
        eye_op = EyesOpenOperator()

        mock_context.config.face_landmarker_url = "http://model.task"
        mock_context.config.models_dir = "models"

        # Mock mediapipe properly
        mock_mp = MagicMock()

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("core.managers.get_face_landmarker") as mock_get_lm,
            patch("core.operators.face_metrics.mp", mock_mp),
        ):
            mock_lm = mock_get_lm.return_value
            mock_res = MagicMock()
            # categories must have category_name and score
            c1 = MagicMock()
            c1.category_name = "eyeBlinkLeft"
            c1.score = 0.1
            mock_res.face_blendshapes = [[c1]]
            mock_lm.detect.return_value = mock_res

            # This should trigger the real _get_face_data logic
            result = eye_op.execute(mock_context)
            assert mock_get_lm.called
            assert mock_lm.detect.called
            assert "eyes_open" in result.metrics

    def test_operators_no_data(self, mock_context):
        eye_op = EyesOpenOperator()
        pose_op = FacePoseOperator()

        # No mock data in params or shared_data

        res1 = eye_op.execute(mock_context)
        assert any("No face blendshapes" in str(w) for w in res1.warnings)

        res2 = pose_op.execute(mock_context)
        assert any("No face transformation matrix" in str(w) for w in res2.warnings)
