from pathlib import Path
from unittest.mock import MagicMock, patch

from core.managers.model_loader import initialize_analysis_models
from core.models import AnalysisParameters


def test_yolo26_model_mapping(tmp_path):
    """Verify that all YOLO26 model versions correctly map to their config URLs."""
    config = MagicMock()
    config.models_dir = tmp_path / "models"
    config.models_dir.mkdir()
    config.face_landmarker_url = "http://fake.url/landmarker.task"
    config.yolo26n_url = "http://fake.url/yolo26n.onnx"
    config.yolo26s_url = "http://fake.url/yolo26s.onnx"
    config.yolo26m_url = "http://fake.url/yolo26m.onnx"
    config.yolo26l_url = "http://fake.url/yolo26l.onnx"
    config.yolo26x_url = "http://fake.url/yolo26x.onnx"
    config.yolo12l_seg_url = "http://fake.url/yolo12l-seg.onnx"

    logger = MagicMock()
    registry = MagicMock()

    yolo_versions = ["YOLO26n", "YOLO26s", "YOLO26m", "YOLO26l", "YOLO26x", "YOLO12l-Seg"]

    for version in yolo_versions:
        params = AnalysisParameters(output_folder=str(tmp_path), video_path="test.mp4", subject_detector_model=version)
        registry.get_subject_detector.reset_mock()

        with (
            patch("core.managers.model_loader.download_model"),
            patch("core.managers.model_loader.get_face_landmarker"),
        ):
            # Mock the landmarker file existence to skip that part
            landmarker_path = config.models_dir / "landmarker.task"
            landmarker_path.touch()

            # Mock the model file existence to trigger PersonDetector init
            model_url = getattr(config, f"{version.lower().replace('-', '_')}_url")
            model_path = config.models_dir / Path(model_url).name
            model_path.touch()

            initialize_analysis_models(params, config, logger, registry)

            # Verify registry.get_subject_detector was called with the correct path for this version
            registry.get_subject_detector.assert_called_with(version, str(model_path), logger, "cpu")
