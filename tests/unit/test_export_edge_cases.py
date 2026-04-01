import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from core.filtering import apply_all_filters_vectorized
from core.operators.crop import calculate_best_crop


class TestExportEdgeCases:
    @given(
        frame_w=st.integers(100, 4096),
        frame_h=st.integers(100, 4096),
        box_x=st.integers(0, 4000),
        box_y=st.integers(0, 4000),
        box_w=st.integers(1, 4000),
        box_h=st.integers(1, 4000),
        padding_factor=st.floats(1.0, 2.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_calculate_best_crop_bounds_safety(self, frame_w, frame_h, box_x, box_y, box_w, box_h, padding_factor):
        """Ensure that calculate_best_crop never returns a box exceeding frame dimensions or excluding subject."""
        # Ensure subject box is valid and within frame for this specific test logic
        box_x = min(box_x, frame_w - 1)
        box_y = min(box_y, frame_h - 1)
        box_w = max(1, min(box_w, frame_w - box_x))
        box_h = max(1, min(box_h, frame_h - box_y))

        subject_bbox = (box_x, box_y, box_w, box_h)
        aspect_ratios = [("1:1", 1.0), ("16:9", 16 / 9), ("9:16", 9 / 16)]

        crop = calculate_best_crop(frame_w, frame_h, subject_bbox, aspect_ratios, padding_factor)

        if crop:
            # Check bounds
            assert crop["x"] >= 0
            assert crop["y"] >= 0
            assert crop["x"] + crop["w"] <= frame_w
            assert crop["y"] + crop["h"] <= frame_h

            # Check subject containment (allowing for small float errors)
            assert crop["x"] <= subject_bbox[0] + 1
            assert crop["y"] <= subject_bbox[1] + 1
            assert crop["x"] + crop["w"] >= subject_bbox[0] + subject_bbox[2] - 1
            assert crop["y"] + crop["h"] >= subject_bbox[1] + subject_bbox[3] - 1

    def test_filtering_nan_quality_scores(self, mock_config):
        """Test filtering handles NaN or None quality scores without crashing."""
        frames = [
            {"filename": "f1.png", "metrics": {"quality_score": float("nan")}, "face_sim": np.nan},
            {"filename": "f2.png", "metrics": {"quality_score": 50.0}, "face_sim": 0.8},
        ]

        filters = {"quality_score_min": 40.0, "face_sim_min": 0.5, "face_sim_enabled": True}

        kept, rejected, counter, reasons = apply_all_filters_vectorized(frames, filters, mock_config)

        assert len(kept) == 1
        assert kept[0]["filename"] == "f2.png"
        assert len(rejected) == 1
        assert rejected[0]["filename"] == "f1.png"

    def test_filtering_extreme_thresholds(self, mock_config):
        """Test filtering with extreme thresholds."""
        frames = [{"filename": "f1.png", "metrics": {"quality_score": 50.0}, "mask_area_pct": 10.0}]

        # 1. Reject everything
        filters = {"quality_score_min": 100.0}
        kept, _, _, _ = apply_all_filters_vectorized(frames, filters, mock_config)
        assert len(kept) == 0

        # 2. Include everything
        filters = {"quality_score_min": 0.0, "mask_area_enabled": True, "mask_area_pct_min": 0.0}
        kept, _, _, _ = apply_all_filters_vectorized(frames, filters, mock_config)
        assert len(kept) == 1
