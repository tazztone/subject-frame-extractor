from unittest.mock import patch

import cv2
import numpy as np

from core.face_clustering import cluster_faces, get_cluster_representative


def test_cluster_faces_empty():
    """Test clustering with no faces."""
    labels, mapping = cluster_faces([])
    assert len(labels) == 0
    assert mapping == {}


@patch("core.face_clustering.DBSCAN")
def test_cluster_faces_logic(mock_dbscan):
    """Test clustering logic with mocked DBSCAN."""
    mock_instance = mock_dbscan.return_value
    mock_instance.labels_ = np.array([0, 0, 1, -1])
    mock_instance.fit.return_value = mock_instance

    faces = [
        {"embedding": np.random.rand(512)},
        {"embedding": np.random.rand(512)},
        {"embedding": np.random.rand(512)},
        {"embedding": np.random.rand(512)},
    ]

    labels, mapping = cluster_faces(faces)

    assert np.array_equal(labels, np.array([0, 0, 1, -1]))
    # Labels are [0, 1] (excluding -1)
    assert mapping == {0: 0, 1: 1}
    mock_dbscan.assert_called_once()


@patch("core.face_clustering.cv2.VideoCapture")
@patch("core.face_clustering.cv2.imread")
@patch("core.face_clustering.cv2.imwrite")
def test_get_cluster_representative(mock_imwrite, mock_imread, mock_video_capture):
    """Test finding and saving a cluster representative."""
    # Setup mocks
    mock_cap = mock_video_capture.return_value
    mock_cap.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))

    mock_imread.return_value = np.zeros((50, 50, 3), dtype=np.uint8)

    faces = [
        {"det_score": 0.8, "frame_num": 10, "bbox": [10, 10, 40, 40], "thumb_path": "t1.jpg"},
        {"det_score": 0.9, "frame_num": 20, "bbox": [5, 5, 35, 35], "thumb_path": "t2.jpg"},
    ]
    labels = np.array([0, 0])

    path, crop, msg = get_cluster_representative(faces, labels, 0, "vid.mp4", "out")

    assert "Selected Person 0" in msg
    assert path is not None
    assert crop is not None
    # Best face was second one (score 0.9)
    mock_cap.set.assert_any_call(cv2.CAP_PROP_POS_FRAMES, 20)


def test_get_cluster_representative_not_found():
    """Test representative selection when cluster label is missing."""
    faces = [{"det_score": 0.8}]
    labels = np.array([1])
    path, crop, msg = get_cluster_representative(faces, labels, 0, "vid.mp4", "out")
    assert "Cluster not found" in msg
    assert path is None


@patch("core.face_clustering.cv2.VideoCapture")
@patch("core.face_clustering.cv2.imread")
def test_get_cluster_representative_thumb_fail(mock_imread, mock_video_capture):
    """Test representative selection when thumbnail read fails."""
    mock_cap = mock_video_capture.return_value
    mock_cap.read.return_value = (True, np.zeros((10, 10, 3)))
    mock_imread.return_value = None
    faces = [{"det_score": 0.8, "frame_num": 10, "thumb_path": "invalid.jpg"}]
    labels = np.array([0])
    path, crop, msg = get_cluster_representative(faces, labels, 0, "vid.mp4", "out")
    assert "Could not read thumbnail" in msg


@patch("core.face_clustering.cv2.VideoCapture")
def test_get_cluster_representative_read_fail(mock_video_capture):
    """Test representative selection when video frame read fails."""
    mock_cap = mock_video_capture.return_value
    mock_cap.read.return_value = (False, None)

    faces = [{"det_score": 0.8, "frame_num": 10}]
    labels = np.array([0])

    path, crop, msg = get_cluster_representative(faces, labels, 0, "vid.mp4", "out")
    assert "Could not read video frame" in msg
