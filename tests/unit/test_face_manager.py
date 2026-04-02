from unittest.mock import MagicMock, patch

import pytest

from core.managers.face import get_face_analyzer, get_face_landmarker


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_registry():
    registry = MagicMock()
    # Mock get_or_load to execute the loader immediately
    registry.get_or_load.side_effect = lambda key, loader: loader()
    return registry


def test_get_face_landmarker_success(mock_logger):
    """Test successful initialization and thread-local caching of FaceLandmarker."""
    mock_detector = MagicMock()

    with patch("mediapipe.tasks.python.vision.FaceLandmarker.create_from_options", return_value=mock_detector):
        # 1. First call in this thread should initialize
        res1 = get_face_landmarker("fake_path.task", mock_logger)
        assert res1 == mock_detector
        mock_logger.info.assert_called_with(
            "Initializing MediaPipe FaceLandmarker for new thread.", component="face_landmarker"
        )

        # 2. Second call in same thread should use cache
        mock_logger.info.reset_mock()
        res2 = get_face_landmarker("fake_path.task", mock_logger)
        assert res2 == mock_detector
        mock_logger.info.assert_not_called()


def test_get_face_landmarker_failure(mock_logger):
    """Test initialization failure of FaceLandmarker."""
    # Clear thread local first to ensure initialization is attempted
    from core.managers.face import thread_local

    if hasattr(thread_local, "face_landmarker_instance"):
        del thread_local.face_landmarker_instance

    with patch("mediapipe.tasks.python.vision.FaceLandmarker.create_from_options", side_effect=Exception("Init Fail")):
        with pytest.raises(RuntimeError, match="Could not initialize MediaPipe face landmarker model"):
            get_face_landmarker("fake_path.task", mock_logger)
        mock_logger.error.assert_called()


def test_get_face_analyzer_success(mock_logger, mock_registry):
    """Test successful initialization of FaceAnalysis on CPU."""
    mock_analyzer = MagicMock()

    with patch("insightface.app.FaceAnalysis", return_value=mock_analyzer) as mock_class:
        res = get_face_analyzer("buffalo_l", "models", (640, 640), mock_logger, mock_registry, device="cpu")
        assert res == mock_analyzer
        mock_class.assert_called_once()
        args, kwargs = mock_class.call_args
        assert kwargs["providers"] == ["CPUExecutionProvider"]


def test_get_face_analyzer_cuda_success(mock_logger, mock_registry):
    """Test successful initialization of FaceAnalysis on CUDA."""
    mock_analyzer = MagicMock()

    with patch("insightface.app.FaceAnalysis", return_value=mock_analyzer) as mock_class:
        res = get_face_analyzer("buffalo_l", "models", (640, 640), mock_logger, mock_registry, device="cuda")
        assert res == mock_analyzer
        args, kwargs = mock_class.call_args
        assert "CUDAExecutionProvider" in kwargs["providers"]


def test_get_face_analyzer_cuda_oom_fallback(mock_logger, mock_registry):
    """Test fallback to CPU when CUDA initialization fails with OOM."""
    mock_analyzer_cpu = MagicMock()

    def side_effect(*args, **kwargs):
        if "CUDAExecutionProvider" in kwargs.get("providers", []):
            raise Exception("out of memory")
        return mock_analyzer_cpu

    with patch("insightface.app.FaceAnalysis", side_effect=side_effect):
        res = get_face_analyzer("buffalo_l", "models", (640, 640), mock_logger, mock_registry, device="cuda")
        assert res == mock_analyzer_cpu
        mock_logger.warning.assert_called_with("CUDA OOM, retrying with CPU...")


def test_get_face_analyzer_generic_failure(mock_logger, mock_registry):
    """Test generic initialization failure of FaceAnalysis."""
    with patch("insightface.app.FaceAnalysis", side_effect=Exception("Generic Fail")):
        with pytest.raises(RuntimeError, match="Could not initialize face analysis model"):
            get_face_analyzer("buffalo_l", "models", (640, 640), mock_logger, mock_registry, device="cpu")


def test_get_face_analyzer_cpu_fallback_failure(mock_logger, mock_registry):
    """Test failure during CPU fallback."""

    def side_effect(*args, **kwargs):
        if "CUDAExecutionProvider" in kwargs.get("providers", []):
            raise Exception("out of memory")
        raise Exception("CPU also failed")

    with patch("insightface.app.FaceAnalysis", side_effect=side_effect):
        with pytest.raises(RuntimeError, match="CPU fallback also failed"):
            get_face_analyzer("buffalo_l", "models", (640, 640), mock_logger, mock_registry, device="cuda")
