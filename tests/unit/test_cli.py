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


@patch("core.cli_commands.execute_extraction")
@patch("core.cli_commands._setup_runtime")
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
    video_path.touch()
    output_dir = tmp_path / "output"

    # Invoke command - source and output are positional now
    result = runner.invoke(cli, ["extract", str(video_path), str(output_dir)])

    assert result.exit_code == 0
    assert "EXTRACTION" in result.output
    assert "Extraction complete" in result.output
    mock_execute.assert_called_once()


@patch("core.cli_commands.execute_extraction")
@patch("core.cli_commands._setup_runtime")
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

    # Invoke command - source and output are positional now
    result = runner.invoke(cli, ["extract", str(folder_path), str(output_dir)])

    assert result.exit_code == 0
    assert "INGESTION" in result.output
    assert "Ingestion complete" in result.output
    mock_execute.assert_called_once()


@patch("core.cli_commands.execute_analysis")
@patch("core.cli_commands.execute_propagation")
@patch("core.cli_commands.execute_pre_analysis")
@patch("core.cli_commands._setup_runtime")
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
    mock_pre.return_value = [{"done": True, "unified_log": "Pre-Analysis Complete", "scenes": []}]
    mock_prop.return_value = [mock_pipeline_result]
    mock_analysis.return_value = [mock_pipeline_result]

    # Create session dir
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    source_path = tmp_path / "video.mp4"
    source_path.touch()

    # Invoke command
    result = runner.invoke(cli, ["analyze", "--session", str(session_dir), "--source", str(source_path)])

    assert result.exit_code == 0
    assert "ANALYSIS" in result.output
    assert "Analysis complete" in result.output


def test_status_non_existent(runner, tmp_path):
    result = runner.invoke(cli, ["status", "--session", str(tmp_path / "ghost")])
    assert result.exit_code != 0


def test_status_exists(runner, tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "frame_map.json").write_text("{}")

    result = runner.invoke(cli, ["status", "--session", str(session_dir)])
    assert result.exit_code == 0
    assert str(session_dir) in result.output
    assert "Extraction complete" in result.output


@patch("core.cli_commands._setup_runtime")
@patch("core.cli_commands.Database")
@patch("core.cli_commands.apply_all_filters_vectorized")
def test_filter(mock_apply, mock_db, mock_setup, runner, tmp_path):
    mock_setup.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
    mock_db.return_value.load_all_metadata.return_value = [{"filename": "f1.png"}]
    mock_apply.return_value = ([], [], MagicMock(), [])

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "metadata.db").touch()

    result = runner.invoke(cli, ["filter", "--session", str(session_dir)])
    assert result.exit_code == 0
    assert "FILTERING" in result.output
    assert "Filtering complete" in result.output


@pytest.mark.parametrize("command", ["extract", "analyze", "status", "filter"])
def test_cli_help_commands(runner, command):
    result = runner.invoke(cli, [command, "--help"])
    assert result.exit_code == 0
    assert "Show this message and exit" in result.output


def test_extract_invalid_source(runner, tmp_path):
    result = runner.invoke(cli, ["extract", "non_existent.mp4", str(tmp_path / "out")])
    # Click returns exit code 2 for path-not-found-as-argument
    assert result.exit_code != 0


@patch("core.cli_commands.execute_analysis")
@patch("core.cli_commands.execute_pre_analysis")
@patch("core.cli_commands._setup_runtime")
@patch("torch.cuda.is_available", return_value=False)
def test_analyze_folder(mock_cuda, mock_setup, mock_pre, mock_analysis, runner, tmp_path, mock_pipeline_result):
    mock_setup.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock())
    mock_pre.return_value = [{"done": True, "unified_log": "Pre Complete", "scenes": []}]
    mock_analysis.return_value = [mock_pipeline_result]

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    source_dir = tmp_path / "images"
    source_dir.mkdir()

    # For folders, Propagation should be skipped
    with patch("core.cli_commands.execute_propagation") as mock_prop:
        result = runner.invoke(cli, ["analyze", "--session", str(session_dir), "--source", str(source_dir)])
        assert result.exit_code == 0
        assert "Mask Propagation (Skipped for Folder)" in result.output
        mock_prop.assert_not_called()
