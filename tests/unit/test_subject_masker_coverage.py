from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Completely mock cv2 to avoid "partially initialized" errors
mock_cv2 = MagicMock()
mock_cv2.Mat = MagicMock
mock_cv2.mat_wrapper = MagicMock()
mock_cv2.resize = MagicMock(return_value=np.zeros((100, 100), dtype=np.uint8))
mock_cv2.INTER_NEAREST = 0
mock_cv2.cvtColor = MagicMock(return_value=np.zeros((100, 100, 3), dtype=np.uint8))
mock_cv2.COLOR_RGB2BGR = 1

# Mock heavy dependencies
with patch.dict(
    "sys.modules",
    {
        "torch": MagicMock(),
        "torchvision": MagicMock(),
        "sam3": MagicMock(),
        "insightface": MagicMock(),
        "mediapipe": MagicMock(),
        "cv2": mock_cv2,
    },
):
    from core.models import AnalysisParameters
    from core.scene_utils.subject_masker import SubjectMasker


class TestSubjectMasker:
    """
    Tests for core/scene_utils/subject_masker.py
    """

    @pytest.fixture
    def mock_dependencies(self):
        config = MagicMock()
        logger = MagicMock()
        thumb_manager = MagicMock()
        model_registry = MagicMock()
        return config, logger, thumb_manager, model_registry

    @pytest.fixture
    def subject_masker(self, mock_dependencies):
        config, logger, thumb_manager, model_registry = mock_dependencies

        # Configure models
        models = {"face_analyzer": MagicMock(), "face_landmarker": MagicMock(), "sam3": MagicMock()}
        model_registry.get_or_load.side_effect = lambda name, **kwargs: models.get(name)

        params = AnalysisParameters(
            output_folder="/tmp",
            video_path="test.mp4",
            primary_seed_strategy="🤖 Automatic",
            enable_face_filter=True,  # Trigger warning path
        )

        # Ensure logger is mocked
        if logger is None:
            logger = MagicMock()

        return SubjectMasker(
            params,
            MagicMock(),
            MagicMock(),
            config,
            logger=logger,
            thumbnail_manager=thumb_manager,
            model_registry=model_registry,
        )

    def test_initialization(self, subject_masker):
        """Test proper initialization."""
        # Models should be initialized (but face_analyzer passed as None in fixture constructor)
        # We manually update it if we want it to be not None, but here we just check attributes.
        assert subject_masker.params is not None

    def test_initialize_tracker(self, subject_masker):
        """Test tracker initialization logic."""
        subject_masker.dam_tracker = None
        subject_masker.params.tracker_model_name = "sam3"

        # Mock registry return
        subject_masker.model_registry.get_tracker.return_value = MagicMock()

        success = subject_masker._initialize_tracker()
        assert success is True
        assert subject_masker.dam_tracker is not None

    def test_run_propagation_no_tracker(self, subject_masker):
        """Test propagation fails if tracker cannot be initialized."""
        subject_masker.model_registry.get_tracker.return_value = None
        result = subject_masker.run_propagation("/tmp", [])
        assert result.get("error") is not None
