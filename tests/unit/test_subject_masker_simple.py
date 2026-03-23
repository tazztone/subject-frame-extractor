import threading
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch

import numpy as np

from core.models import AnalysisParameters
from core.scene_utils.subject_masker import SubjectMasker


def test_subject_masker_init():
    params = AnalysisParameters()
    config = MagicMock()
    logger = MagicMock()
    pq = Queue()
    ce = threading.Event()

    with patch.object(SubjectMasker, "initialize_models"):
        masker = SubjectMasker(params, pq, ce, config, logger=logger)
        assert masker.config == config
        assert masker.logger == logger


def test_subject_masker_draw_bbox():
    params = AnalysisParameters()
    config = MagicMock()
    config.visualization_bbox_color = [255, 0, 0]
    config.visualization_bbox_thickness = 2

    masker = SubjectMasker(params, Queue(), threading.Event(), config, logger=MagicMock())

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # The draw_bbox method calls core.image_utils.draw_bbox
    res = masker.draw_bbox(img, [10, 10, 20, 20], color=(255, 0, 0))
    assert not np.array_equal(res, img)


def test_subject_masker_get_thumb_for_frame():
    params = AnalysisParameters(thumbnails_only=True)
    config = MagicMock()
    masker = SubjectMasker(params, Queue(), threading.Event(), config, logger=MagicMock())
    masker.frame_map = {10: "frame_00010.webp"}
    masker.thumbnail_manager = MagicMock()
    masker.thumbnail_manager.get.return_value = np.zeros((10, 10, 3))

    thumb = masker._get_thumb_for_frame(Path("thumbs"), 10)
    assert thumb is not None
    assert masker.thumbnail_manager.get.called


def test_subject_masker_create_frame_map():
    params = AnalysisParameters(thumbnails_only=True)
    masker = SubjectMasker(params, Queue(), threading.Event(), MagicMock(), logger=MagicMock())
    with patch("core.scene_utils.subject_masker.create_frame_map", return_value={1: "f1.webp"}) as mock_cfm:
        fmap = masker._create_frame_map("out")
        assert fmap == {1: "f1.webp"}
        assert mock_cfm.called


def test_subject_masker_load_shot_frames():
    params = AnalysisParameters(thumbnails_only=True)
    masker = SubjectMasker(params, Queue(), threading.Event(), MagicMock(), logger=MagicMock())
    masker.frame_map = {10: "f10.webp", 11: "f11.webp"}
    masker.thumbnail_manager = MagicMock()
    masker.thumbnail_manager.get.return_value = np.zeros((10, 20, 3))

    frames = masker._load_shot_frames("dir", Path("thumbs"), 10, 12)
    assert len(frames) == 2
    assert frames[0][0] == 10
    assert frames[0][2] == (10, 20)
