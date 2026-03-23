from unittest.mock import MagicMock, patch

from core.system_health import (
    check_dependencies,
    check_environment,
    check_paths_and_assets,
    generate_full_diagnostic_report,
)


def test_check_environment():
    """Test environment check logic."""
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.get_device_name", return_value="NVIDIA RTX"):
            report = check_environment()
            assert any("Python Version: OK" in line for line in report)
            assert any("PyTorch Version: OK" in line for line in report)
            assert any("CUDA: OK" in line for line in report)

    with patch("torch.cuda.is_available", return_value=False):
        report = check_environment()
        assert any("CUDA: NOT AVAILABLE" in line for line in report)


def test_check_dependencies():
    """Test dependency check logic."""
    report = check_dependencies()
    # Assuming standard libs are installed in test environment
    assert any("cv2: OK" in line for line in report)


@patch("core.system_health.shutil.which")
@patch("core.system_health.subprocess.run")
def test_check_paths_and_assets(mock_run, mock_which):
    """Test paths and assets check logic."""
    mock_which.return_value = "/usr/bin/exiftool"
    mock_run.return_value.stdout = "12.34"

    config = MagicMock()
    config.models_dir = "models"

    with patch("core.system_health.Path.exists", return_value=True):
        report = check_paths_and_assets(config)
        assert any("ExifTool: OK" in line for line in report)
        assert any("Models Directory: OK" in line for line in report)


@patch("core.system_health.execute_extraction")
@patch("core.system_health.execute_pre_analysis")
@patch("core.system_health.execute_propagation")
@patch("core.system_health.execute_analysis")
@patch("core.system_health.export_kept_frames")
def test_simulate_pipeline(mock_export, mock_ana, mock_prop, mock_pre, mock_ext):
    """Test the E2E pipeline simulation logic."""
    from core.system_health import simulate_pipeline

    config = MagicMock()
    config.downloads_dir = "downloads"
    logger = MagicMock()

    # Mock return values for generators (execute_* return generators)
    mock_ext.return_value = [{"done": True, "extracted_frames_dir_state": "out", "extracted_video_path_state": "v.mp4"}]
    mock_pre.return_value = [{"done": True, "output_dir": "out", "scenes": []}]
    mock_prop.return_value = [{"done": True}]
    mock_ana.return_value = [{"done": True, "output_dir": "out"}]
    mock_export.return_value = "✅ Exported 0 frames"

    # Mock filtering functions used in simulate_pipeline
    with patch("core.filtering.load_and_prep_filter_data", return_value=([], {})):
        with patch("core.filtering.apply_all_filters_vectorized", return_value=([], [], [], [])):
            with patch("core.system_health.shutil.rmtree"):
                with patch("core.system_health.Path.mkdir"):
                    report = simulate_pipeline(config, logger, None, None, None, True)
                    print(f"DEBUG REPORT: {report}")
                    assert any("OK" in line for line in report if "Stage" in line)


def test_simulate_pipeline_failure(logger=MagicMock()):
    """Test simulate_pipeline failure handling."""
    from core.system_health import simulate_pipeline

    config = MagicMock()

    with patch("core.system_health.execute_extraction", side_effect=Exception("Crash")):
        with patch("core.system_health.shutil.rmtree"):
            with patch("core.system_health.Path.mkdir"):
                report = simulate_pipeline(config, MagicMock(), None, None, None, True)
                assert any("FAILED" in line for line in report)


def test_generate_full_diagnostic_report():
    """Test full diagnostic report generation (generator)."""
    config = MagicMock()
    logger = MagicMock()
    progress_queue = MagicMock()
    cancel_event = MagicMock()
    thumbnail_manager = MagicMock()

    # Mock all internal checks to avoid side effects
    with patch("core.system_health.check_environment", return_value=["env ok"]):
        with patch("core.system_health.check_dependencies", return_value=["dep ok"]):
            with patch("core.system_health.check_paths_and_assets", return_value=["path ok"]):
                with patch("core.system_health.simulate_pipeline", return_value=["sim ok"]):
                    report_gen = generate_full_diagnostic_report(
                        config, logger, progress_queue, cancel_event, thumbnail_manager, True
                    )
                    report = next(report_gen)
                    assert "--- System Diagnostics Report ---" in report
                    assert "env ok" in report
                    assert "sim ok" in report
