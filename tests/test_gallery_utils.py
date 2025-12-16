import pytest
from unittest.mock import MagicMock
from ui.gallery_utils import scene_caption, create_scene_thumbnail_with_badge
from core.models import Scene
import numpy as np

def test_scene_caption_dict():
    scene = {'shot_id': 1, 'start_frame': 10, 'end_frame': 20, 'status': 'included', 'seed_type': 'test', 'rejection_reasons': []}
    cap = scene_caption(scene)
    assert "Scene 1" in cap
    assert "10-20" in cap

def test_scene_caption_obj():
    scene = MagicMock(spec=Scene)
    scene.shot_id = 1
    scene.start_frame = 10
    scene.end_frame = 20
    scene.status = 'included'
    scene.rejection_reasons = []
    scene.seed_type = None
    cap = scene_caption(scene)
    assert "Scene 1" in cap

def test_create_scene_thumbnail_with_badge():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = create_scene_thumbnail_with_badge(img, 1, True)
    assert res.shape == img.shape
    assert not np.array_equal(res, img)

def test_create_scene_thumbnail_included():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = create_scene_thumbnail_with_badge(img, 1, False)
    assert res.shape == img.shape
