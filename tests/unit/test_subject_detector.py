from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.managers.subject_detector import SubjectDetector


@pytest.fixture
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_ort_session():
    session = MagicMock()
    # Mock inputs/outputs for a segmentation model by default
    # YOLO-seg: [n_batches, 4+80+32, 8400]
    input_mock = MagicMock()
    input_mock.name = "images"
    input_mock.shape = [1, 3, 640, 640]
    session.get_inputs.return_value = [input_mock]
    out0 = MagicMock()
    out0.name = "output0"
    out0.shape = [1, 116, 100]  # 4 + 80 + 32
    out1 = MagicMock()
    out1.name = "output1"
    out1.shape = [1, 32, 160, 160]
    session.get_outputs.return_value = [out0, out1]
    return session


def test_subject_detector_init(mock_logger, mock_ort_session):
    with patch("core.managers.subject_detector.ort.InferenceSession", return_value=mock_ort_session):
        detector = SubjectDetector("dummy.onnx", mock_logger)
        assert detector.is_seg is True
        assert detector.total_dims == 116
        mock_logger.success.assert_called()


def test_detect_filters_by_class_and_conf(mock_logger, mock_ort_session):
    # preds: (1, 116, 2)
    preds = np.zeros((1, 116, 2), dtype=np.float32)
    # Det 0
    preds[0, :4, 0] = [320, 320, 100, 100]
    preds[0, 4, 0] = 0.9  # Class 0 score
    # Det 1
    preds[0, :4, 1] = [100, 100, 50, 50]
    preds[0, 5, 1] = 0.8  # Class 1 score
    proto = np.zeros((1, 32, 160, 160), dtype=np.float32)
    mock_ort_session.run.return_value = [preds, proto]
    with patch("core.managers.subject_detector.ort.InferenceSession", return_value=mock_ort_session):
        detector = SubjectDetector("dummy.onnx", mock_logger)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Detect person (class 0)
        results = detector.detect(frame, target_class_id=0, conf_threshold=0.5)
        assert len(results) == 1
        assert results[0].class_id == 0
        assert results[0].conf == pytest.approx(0.9)
        assert results[0].type == "yolo12l_seg"


def test_postprocess_det_unscaling(mock_logger, mock_ort_session):
    # Test non-seg model
    mock_ort_session.get_outputs.return_value = [mock_ort_session.get_outputs()[0]]
    # preds: (1, 84, 1)
    preds = np.zeros((1, 84, 1), dtype=np.float32)
    preds[0, :4, 0] = [320, 320, 100, 100]
    preds[0, 4, 0] = 0.9
    mock_ort_session.run.return_value = [preds]
    with patch("core.managers.subject_detector.ort.InferenceSession", return_value=mock_ort_session):
        detector = SubjectDetector("dummy.onnx", mock_logger)
        assert detector.is_seg is False
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        results = detector.detect(frame)
        assert len(results) == 1
        assert results[0].bbox == [540, 260, 740, 460]


def test_postprocess_seg_mask_logic(mock_logger, mock_ort_session):
    # preds: (1, 116, 1)
    preds = np.zeros((1, 116, 1), dtype=np.float32)
    preds[0, :4, 0] = [320, 320, 640, 640]
    preds[0, 4, 0] = 1.0
    preds[0, 84, 0] = 10.0
    # proto: (1, 32, 160, 160)
    proto = np.zeros((1, 32, 160, 160), dtype=np.float32)
    proto[0, 0, :, :] = 1.0
    mock_ort_session.run.return_value = [preds, proto]
    with patch("core.managers.subject_detector.ort.InferenceSession", return_value=mock_ort_session):
        detector = SubjectDetector("dummy.onnx", mock_logger)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        results = detector.detect(frame)
        assert len(results) == 1
        assert results[0].mask is not None
        assert results[0].mask.shape == (720, 1280)
        assert results[0].mask.any()


def test_dynamic_class_splitting(mock_logger, mock_ort_session):
    out0 = MagicMock()
    out0.shape = [1, 30, 10]
    out1 = MagicMock()
    out1.shape = [1, 16, 160, 160]
    mock_ort_session.get_outputs.return_value = [out0, out1]
    preds = np.zeros((1, 30, 1), dtype=np.float32)
    preds[0, :4, 0] = [320, 320, 100, 100]
    preds[0, 13, 0] = 0.9
    preds[0, 14:, 0] = 5.0
    proto = np.zeros((1, 16, 160, 160), dtype=np.float32)
    mock_ort_session.run.return_value = [preds, proto]
    with patch("core.managers.subject_detector.ort.InferenceSession", return_value=mock_ort_session):
        detector = SubjectDetector("dummy.onnx", mock_logger)
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        results = detector.detect(frame, target_class_id=9)
        assert len(results) == 1
        assert results[0].class_id == 9


def test_close_releases_session(mock_logger, mock_ort_session):
    with patch("core.managers.subject_detector.ort.InferenceSession", return_value=mock_ort_session):
        detector = SubjectDetector("dummy.onnx", mock_logger)
        detector.close()
