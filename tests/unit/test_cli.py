import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

# Ensure project root is in path and comes before site-packages
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cli import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_pipeline_result():
    return {"done": True, "unified_log": "Success"}


def test_cli_version(runner):
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "Subject Frame Extractor CLI" in result.output


@patch("cli.execute_extraction")
@patch("cli._setup_runtime")
def test_extract_video(mock_setup, mock_execute, runner, tmp_path, mock_pipeline_result):
    # Setup mocks
    mock_setup.return_value = (
        MagicMock(),  # config
        MagicMock(),  # logger
        MagicMock(),  # progress_queue
        MagicMock(),  # cancel_event
        MagicMock(),  # model_registry
        MagicMock(),  # thumbnail_manager
    )
    mock_execute.return_value = [mock_pipeline_result]

    # Create a dummy video file
    video_path = tmp_path / "test.mp4"
    video_path.write_text("dummy")
    output_dir = tmp_path / "output"

    # Invoke command
    result = runner.invoke(cli, ["extract", "--source", str(video_path), "--output", str(output_dir)])

    assert result.exit_code == 0
    assert "EXTRACTION" in result.output
    assert "Extraction complete" in result.output
    mock_execute.assert_called_once()


@patch("cli.execute_extraction")
@patch("cli._setup_runtime")
def test_extract_folder(mock_setup, mock_execute, runner, tmp_path, mock_pipeline_result):
    # Setup mocks
    mock_setup.return_value = (
        MagicMock(),  # config
        MagicMock(),  # logger
        MagicMock(),  # progress_queue
        MagicMock(),  # cancel_event
        MagicMock(),  # model_registry
        MagicMock(),  # thumbnail_manager
    )
    mock_execute.return_value = [mock_pipeline_result]

    # Create a dummy folder
    folder_path = tmp_path / "images"
    folder_path.mkdir()
    output_dir = tmp_path / "output"

    # Invoke command
    result = runner.invoke(cli, ["extract", "--source", str(folder_path), "--output", str(output_dir)])

    assert result.exit_code == 0
    assert "INGESTION" in result.output
    assert "Ingestion complete" in result.output
    mock_execute.assert_called_once()


@patch("cli.execute_analysis")
@patch("cli.execute_propagation")
@patch("cli.execute_pre_analysis")
@patch("cli._setup_runtime")
@patch("torch.cuda.is_available", return_value=False)
def test_analyze(mock_cuda, mock_setup, mock_pre, mock_prop, mock_analysis, runner, tmp_path, mock_pipeline_result):
    mock_setup.return_value = (
        MagicMock(),  # config
        MagicMock(),  # logger
        MagicMock(),  # progress_queue
        MagicMock(),  # cancel_event
        MagicMock(),  # model_registry
        MagicMock(),  # thumbnail_manager
    )
    mock_pre.return_value = [{"done": True, "scenes": [{"scene_id": 1, "start_frame": 0, "end_frame": 10}]}]
    mock_prop.return_value = [mock_pipeline_result]
    mock_analysis.return_value = [mock_pipeline_result]

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    source_path = tmp_path / "test.mp4"
    source_path.write_text("dummy")

    result = runner.invoke(cli, ["analyze", "--session", str(session_dir), "--source", str(source_path)])

    assert result.exit_code == 0
    assert "ANALYSIS" in result.output
    assert "Stage 1: Pre-Analysis" in result.output
    assert "Stage 2: Mask Propagation" in result.output
    assert "Stage 3: Metric Analysis" in result.output

    mock_pre.assert_called_once()
    mock_prop.assert_called_once()
    mock_analysis.assert_called_once()


def test_status_non_existent(runner, tmp_path):
    result = runner.invoke(cli, ["status", "--session", str(tmp_path / "non_existent")])
    assert result.exit_code != 0


@patch("core.fingerprint.load_fingerprint")
def test_status_exists(mock_load_fp, runner, tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "frame_map.json").write_text("{}")

    mock_fp = MagicMock()
    mock_fp.created_at = "2023-01-01"
    mock_fp.video_path = "test.mp4"
    mock_fp.video_size = 1000
    mock_load_fp.return_value = mock_fp

    result = runner.invoke(cli, ["status", "--session", str(session_dir)])
    assert result.exit_code == 0
    assert "SESSION STATUS" in result.output
    assert "Extraction complete" in result.output


@patch("cli.apply_all_filters_vectorized")
@patch("cli.Database")
@patch("cli._setup_runtime")
def test_filter(mock_setup, mock_db, mock_filter, runner, tmp_path):
    mock_setup.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    db_path = session_dir / "metadata.db"
    db_path.write_text("dummy")

    mock_db_instance = mock_db.return_value
    mock_db_instance.load_all_metadata.return_value = [{"id": 1}]

    mock_filter.return_value = (
        [{"id": 1}],  # kept
        [],  # rejected
        MagicMock(),  # rejection_counts
        [],  # reasons
    )

    result = runner.invoke(cli, ["filter", "--session", str(session_dir), "--quality-min", "50"])
    assert result.exit_code == 0
    assert "FILTERING" in result.output
    assert "Kept:     1" in result.output
