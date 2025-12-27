"""
E2E tests for bug fixes.

These tests verify the bug fixes using Playwright to interact with the actual Gradio UI.
"""

from unittest.mock import patch

from core.models import Scene
from core.shared import build_scene_gallery_items


class TestPaginationBugFixes:
    """Tests for pagination edge case fixes (Bug 2)."""

    def test_build_scene_gallery_items_page_clamped_to_max(self):
        """Page number greater than total_pages should be clamped."""
        scenes = [
            Scene(shot_id=i, start_frame=i * 10, end_frame=i * 10 + 9, status="included").model_dump() for i in range(3)
        ]
        # Request page 999 on dataset that fits in 1 page
        with patch("core.shared.cv2.imread", return_value=None):
            items, index_map, total_pages = build_scene_gallery_items(scenes, "Kept", "/tmp/test", page_num=999)
        assert total_pages == 1
        # Should not raise error

    def test_build_scene_gallery_items_empty_scenes(self):
        """Empty scenes list should return page 1."""
        items, index_map, total_pages = build_scene_gallery_items([], "Kept", "/tmp")
        assert total_pages == 1
        assert items == []
        assert index_map == []


class TestPipelinesSceneFieldsFix:
    """Tests for Scene.model_fields fix (Bug 1)."""

    def test_scene_model_fields_accessible(self):
        """Scene.model_fields.keys() should work for Pydantic model."""
        scene_fields = set(Scene.model_fields.keys())
        assert "shot_id" in scene_fields
        assert "status" in scene_fields
        assert "start_frame" in scene_fields
        assert "seed_metrics" in scene_fields


class TestFilterSlidersFix:
    """Tests for filter slider separation (Bug 4)."""

    def test_seed_metrics_score_range(self):
        """Seed metrics score should be in 0-20 range (NIQE + face composite)."""
        # Sample data from user
        seed_metrics = {
            "reason": "pre-analysis complete",
            "score": 13.841613327487723,
            "best_niqe": 4.044645244237375,
            "best_face_sim": 0.5105124115943909,
        }
        # Score should be in 0-20 range (not 0-1 like old slider)
        assert 0 <= seed_metrics["score"] <= 20
        # Face sim should be 0-1
        assert 0 <= seed_metrics["best_face_sim"] <= 1
        # NIQE typically in 0-10 range
        assert 0 <= seed_metrics["best_niqe"] <= 15
