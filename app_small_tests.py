import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

import app_small as app


def test_sanitize_filename():
    assert app.sanitize_filename("weird/name?file.mp4") == "weird_name_file.mp4"
    assert app.sanitize_filename("a" * 100, max_length=10) == "a" * 10


def test_safe_execute_with_retry(monkeypatch):
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("fail")
        return "ok"

    # Avoid real sleeps
    monkeypatch.setattr(app.time, "sleep", lambda *_: None)

    out = app.safe_execute_with_retry(flaky, max_retries=3, delay=0.01, backoff=1.0)
    assert out == "ok"
    assert calls["n"] == 3


def test_compute_entropy_uniform():
    hist = np.ones(256, dtype=np.float64)
    e = app.compute_entropy(hist)
    assert 0.95 <= e <= 1.0


def test_compute_edge_strength_zero():
    sob = np.zeros((10, 10), dtype=np.float64)
    assert app.compute_edge_strength(sob, sob) == 0


def test_frame_quality_metrics_basic():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (20, 20), (80, 80), (255, 255, 255), -1)
    f = app.Frame(img, frame_number=1)
    f.calculate_quality_metrics(mask=None)
    m = f.metrics
    for v in [
        m.quality_score,
        m.sharpness_score,
        m.edge_strength_score,
        m.contrast_score,
        m.brightness_score,
        m.entropy_score,
    ]:
        assert 0.0 <= v <= 100.0
    assert f.error is None


def test_frame_quality_metrics_small_mask_sets_error():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:9, :11] = 255  # 99 pixels active
    f = app.Frame(img, frame_number=2)
    f.calculate_quality_metrics(mask=mask)
    assert f.error and "Mask too small" in f.error


def test_videomanager_prepare_video_nonexistent_local():
    vm = app.VideoManager("/path/does/not/exist.mp4")
    with pytest.raises(FileNotFoundError):
        vm.prepare_video()


def test_check_dependencies_raises_when_missing_ffmpeg(monkeypatch):
    monkeypatch.setattr(app.shutil, "which", lambda *_: None)
    with pytest.raises(RuntimeError):
        app.check_dependencies()


def test_to_json_safe_rounds_nested(tmp_path):
    data = {"a": 1.123456, "b": [0.3333333, {"c": 2.987654}], "p": tmp_path}
    out = app._to_json_safe(data)
    assert out["a"] == 1.1235
    assert out["b"][0] == 0.3333
    assert out["b"][1]["c"] == 2.9877
    assert out["p"] == str(tmp_path)


def test_create_frame_map_sequential_fallback(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for i in [3, 5, 7]:
        cv2.imwrite(str(frames_dir / f"frame_{i:06d}.png"), np.zeros((10, 10, 3), dtype=np.uint8))

    params = app.AnalysisParameters(output_folder=str(frames_dir), source_path="")
    pipe = app.AnalysisPipeline(params, app.Queue(), app.threading.Event())
    fmap = pipe._create_frame_map()

    assert fmap[3] == "frame_000003.png"
    assert fmap[5] == "frame_000005.png"
    assert fmap[7] == "frame_000007.png"


def test_appui_crop_frame(tmp_path):
    ui = app.AppUI()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (60, 60), (255, 255, 255), -1)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[40:60, 40:60] = 255
    cropped = ui._crop_frame(img, mask, "1:1", 0)
    assert cropped.shape[0] == cropped.shape[1]
    assert cropped.shape[0] > 0


def test_vectorized_filtering_overall_like(tmp_path):
    # Build a small metadata.jsonl
    md = tmp_path / "metadata.jsonl"
    header = {"params": {"enable_face_filter": False}}
    frames = [
        {
            "filename": "frame_000001.png",
            "metrics": {
                "sharpness_score": 60,
                "edge_strength_score": 50,
                "contrast_score": 50,
                "brightness_score": 50,
                "entropy_score": 50,
            },
        },
        {
            "filename": "frame_000002.png",
            "metrics": {
                "sharpness_score": 10,
                "edge_strength_score": 10,
                "contrast_score": 10,
                "brightness_score": 10,
                "entropy_score": 10,
            },
        },
    ]
    with md.open("w") as f:
        f.write(json.dumps(header) + "\n")
        for fr in frames:
            f.write(json.dumps(fr) + "\n")

    ui = app.AppUI()
    all_frames, _ = ui.load_and_prep_filter_data(str(md))

    # Emulate an OVERALL-like threshold: per-metric >= 40
    filters = {}
    for k in app.config.QUALITY_METRICS:
        filters[f"{k}_min"] = 40
        filters[f"{k}_max"] = 100
    filters["enable_dedup"] = False

    kept, rejected, counts, reasons = ui._apply_all_filters_vectorized(all_frames, filters)
    assert len(kept) == 1 and kept[0]["filename"] == "frame_000001.png"


def test_apply_filters_and_export(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "masks").mkdir()

    img1 = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.imwrite(str(out_dir / "frame_000001.png"), img1)
    img2 = np.zeros((50, 50, 3), dtype=np.uint8)
    cv2.imwrite(str(out_dir / "frame_000002.png"), img2)

    # Optional mask for first image (not strictly required for this test)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:40, 10:40] = 255
    cv2.imwrite(str(out_dir / "masks" / "frame_000001.png"), mask)

    md = out_dir / "metadata.jsonl"
    header = {"params": {"enable_face_filter": False}}
    entry1 = {
        "filename": "frame_000001.png",
        "metrics": {
            "sharpness_score": 100,
            "edge_strength_score": 100,
            "contrast_score": 100,
            "brightness_score": 100,
            "entropy_score": 100,
        },
        "mask_path": "frame_000001.png",
        "mask_empty": False,
    }
    entry2 = {
        "filename": "frame_000002.png",
        "metrics": {
            "sharpness_score": 0,
            "edge_strength_score": 0,
            "contrast_score": 0,
            "brightness_score": 0,
            "entropy_score": 0,
        },
        "mask_empty": True,
    }
    with md.open("w") as f:
        f.write(json.dumps(header) + "\n")
        f.write(json.dumps(entry1) + "\n")
        f.write(json.dumps(entry2) + "\n")

    ui = app.AppUI()
    # Stub components for headless test
    ui.components = {}
    ui.components['metric_sliders'] = {}
    for k in app.config.QUALITY_METRICS:
        ui.components['metric_sliders'][f"{k}_min"] = None
        ui.components['metric_sliders'][f"{k}_max"] = None

    all_frames, _ = ui.load_and_prep_filter_data(str(md))

    # Export with sliders set to keep scores >= 10 for all metrics
    # Order: max,min for each metric in alphabetical order (as sorted keys)
    slider_values = []
    for _ in app.config.QUALITY_METRICS:
        slider_values.extend([100, 10])  # max, min

    msg = ui.export_kept_frames(
        all_frames,
        str(out_dir),
        False,  # enable_crop
        "1:1",  # crop_ars
        0,      # crop_padding
        False,  # require_face_match
        5,      # dedup_thresh
        *slider_values
    )
    assert "Exported" in msg and "/2" in msg

    exported_dir = next(
        p for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("out_exported_")
    )
    assert (exported_dir / "frame_000001.png").exists()


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-q"]))
