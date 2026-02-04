"""
Tests for shared utilities in core/shared.py.
"""

from unittest.mock import patch

import numpy as np

from core.models import Scene
from core.shared import build_scene_gallery_items, create_scene_thumbnail_with_badge, scene_caption, scene_matches_view


class TestSharedUtils:
    def test_scene_matches_view(self):
        scene_inc = Scene(shot_id=1, start_frame=0, end_frame=1, status="included")
        scene_exc = Scene(shot_id=2, start_frame=2, end_frame=3, status="excluded")

        # View "All"
        assert scene_matches_view(scene_inc, "All")
        assert scene_matches_view(scene_exc, "All")

        # View "Kept"
        assert scene_matches_view(scene_inc, "Kept")
        assert not scene_matches_view(scene_exc, "Kept")

        # View "Rejected"
        assert not scene_matches_view(scene_inc, "Rejected")
        assert scene_matches_view(scene_exc, "Rejected")

    def test_create_scene_thumbnail_with_badge(self):
        # Create a simple green image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:] = (0, 255, 0)

        # Non-excluded scene: should be same as input (copy)
        res_ok = create_scene_thumbnail_with_badge(img, 1, False)
        assert np.array_equal(res_ok, img)

        # Excluded scene: should have modifications (badge)
        res_bad = create_scene_thumbnail_with_badge(img, 2, True)
        assert not np.array_equal(res_bad, img)

        # Check if badge text "E" is likely drawn (simple check: image changed significantly)
        # Or check if specific pixels are changed.
        # The function draws a rectangle border and a circle badge.
        # Border is 4px thick. (0,0) should be teal (33, 128, 141) in BGR?
        # Wait, CV2 uses BGR default, but inputs are usually RGB in this app?
        # The function docstring says "RGB thumbnail image".
        # Border color is (33, 128, 141).
        # Let's check pixel (0,0)
        assert np.any(res_bad[0, 0] != [0, 255, 0])

    def test_scene_caption(self):
        scene = Scene(shot_id=5, start_frame=10, end_frame=20, status="included", seed_type="manual")
        cap = scene_caption(scene)
        assert "Scene 5" in cap
        assert "[10-20]" in cap
        assert "✅" in cap
        assert "Seed: manual" in cap

        scene_rej = Scene(shot_id=6, start_frame=30, end_frame=40, status="excluded", rejection_reasons=["blurry"])
        cap_rej = scene_caption(scene_rej)
        assert "❌" in cap_rej
        assert "(blurry)" in cap_rej

    @patch("cv2.imread")
    def test_build_scene_gallery_items(self, mock_imread, tmp_path):
        output_dir = tmp_path / "output"
        previews_dir = output_dir / "previews"
        previews_dir.mkdir(parents=True)

        # Create dummy thumbnail file
        (previews_dir / "scene_00001.jpg").touch()
        (previews_dir / "scene_00002.jpg").touch()

        # Mock image loading
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_imread.return_value = img

        scenes = [
            Scene(shot_id=1, start_frame=0, end_frame=10, status="included"),
            Scene(shot_id=2, start_frame=11, end_frame=20, status="excluded"),
        ]

        # Test pagination and filtering
        # View All, Page 1
        items, index_map, total_pages = build_scene_gallery_items(
            scenes, "All", str(output_dir), page_num=1, page_size=10
        )
        assert len(items) == 2
        assert len(index_map) == 2
        assert total_pages == 1

        # View Kept
        items, index_map, total_pages = build_scene_gallery_items(
            scenes, "Kept", str(output_dir), page_num=1, page_size=10
        )
        assert len(items) == 1
        assert index_map == [0]

        # Pagination
        items, index_map, total_pages = build_scene_gallery_items(
            scenes, "All", str(output_dir), page_num=1, page_size=1
        )
        assert len(items) == 1
        assert total_pages == 2
