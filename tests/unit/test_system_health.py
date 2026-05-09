"""
Tests for system health diagnostics.
"""

from unittest.mock import MagicMock, patch

from core.system_health import check_dependencies, check_environment, check_paths_and_assets


def test_check_environment():
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_name", return_value="Test GPU"),
        patch("torch.version.cuda", "12.1"),
    ):
        report = check_environment()
        assert any("CUDA: OK" in line for line in report)
        assert any("CUDA: OK" in line for line in report)
        assert any("Test GPU" in line for line in report)


def test_check_environment_no_cuda():
    with patch("torch.cuda.is_available", return_value=False):
        report = check_environment()
        assert any("Running in CPU mode" in line for line in report)
        assert any("Running in CPU mode" in line for line in report)


def test_check_dependencies():
    report = check_dependencies()
    assert any("gradio: OK" in line for line in report)


def test_check_paths_and_assets():
    config = MagicMock()
    config.models_dir = "models"
    with patch("shutil.which", return_value="/usr/bin/exiftool"), patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "12.30"
        report = check_paths_and_assets(config)
        assert any("ExifTool: OK" in line for line in report)


def test_generate_full_diagnostic_report():
    config = MagicMock()
    config.models_dir = "models"
    logger = MagicMock()
    queue = MagicMock()
    cancel = MagicMock()
    tm = MagicMock()
    # Mock simulate_pipeline to avoid recursion/heavy work
    with (
        patch("core.system_health.simulate_pipeline", return_value=["Pipeline: OK"]),
        patch("shutil.which", return_value=None),
    ):
        from core.system_health import generate_full_diagnostic_report

        gen = generate_full_diagnostic_report(config, logger, queue, cancel, tm, False)
        report = next(gen)
        assert "--- System Diagnostics Report ---" in report
        assert "Pipeline: OK" in report


def test_check_environment_torch_exception():
    with patch("torch.cuda.is_available", side_effect=Exception("Torch Crash")):
        from core.system_health import check_environment

        report = check_environment()
        assert any("PyTorch/CUDA Check: FAILED" in line for line in report)


def test_simulate_pipeline_success(tmp_path):
    config = MagicMock()
    config.downloads_dir = str(tmp_path / "downloads")
    config.default_tracker_model_name = "sam2"
    logger = MagicMock()
    queue = MagicMock()
    cancel = MagicMock()
    tm = MagicMock()

    out_dir_path = tmp_path / "out"
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_dir = str(out_dir_path)

    video_path_obj = tmp_path / "v.mp4"
    video_path_obj.write_text("dummy")
    video_path = str(video_path_obj)

    # Mocking all execute_* functions in core.system_health
    with (
        patch(
            "core.system_health.execute_extraction",
            return_value=[
                {"done": True, "extracted_frames_dir_state": out_dir, "extracted_video_path_state": video_path}
            ],
        ),
        patch(
            "core.system_health.execute_pre_analysis",
            return_value=[{"done": True, "scenes": [{"shot_id": 1, "status": "included"}], "output_dir": out_dir}],
        ),
        patch("core.system_health.execute_propagation", return_value=[{"done": True, "output_dir": out_dir}]),
        patch("core.system_health.execute_analysis", return_value=[{"done": True, "output_dir": out_dir}]),
        patch("core.system_health.export_kept_frames", return_value="Export successful"),
        patch("core.filtering.load_and_prep_filter_data", return_value=([{"f": 1}], {})),
        patch("core.filtering.apply_all_filters_vectorized", return_value=([{"f": 1}], [], [], [])),
        patch("shutil.rmtree"),
        patch("pathlib.Path.mkdir"),
        patch("pathlib.Path.exists", return_value=True),
    ):
        from core.system_health import simulate_pipeline

        report = simulate_pipeline(config, logger, queue, cancel, tm, False)

        # Verify that we got OK for all stages
        stages_covered = [line for line in report if "Stage" in line and "OK" in line]
        assert len(stages_covered) >= 5
