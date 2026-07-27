from unittest.mock import patch
import pytest
from click.testing import CliRunner

from core.cli_args import cli

@pytest.fixture
def runner():
    return CliRunner()

@patch("core.cli_args.run_extract")
def test_extract_command(mock_run, runner, tmp_path):
    source = tmp_path / "test.mp4"
    source.touch()
    out = tmp_path / "out"

    result = runner.invoke(cli, ["extract", str(source), str(out)])
    assert result.exit_code == 0
    mock_run.assert_called_once()

@patch("core.cli_args.run_analyze")
def test_analyze_command(mock_run, runner, tmp_path):
    source = tmp_path / "test.mp4"
    source.touch()
    session = tmp_path / "session"
    session.mkdir()

    result = runner.invoke(cli, ["analyze", "-s", str(session), "-v", str(source)])
    assert result.exit_code == 0
    mock_run.assert_called_once()

@patch("core.cli_args.run_full")
def test_full_command(mock_run, runner, tmp_path):
    source = tmp_path / "test.mp4"
    source.touch()
    out = tmp_path / "out"

    result = runner.invoke(cli, ["full", "-v", str(source), "-o", str(out)])
    assert result.exit_code == 0
    mock_run.assert_called_once()

@patch("core.cli_args.run_status")
def test_status_command(mock_run, runner, tmp_path):
    session = tmp_path / "session"
    session.mkdir()

    result = runner.invoke(cli, ["status", "-s", str(session)])
    assert result.exit_code == 0
    mock_run.assert_called_once()

@patch("core.cli_args.run_filter")
def test_filter_command(mock_run, runner, tmp_path):
    session = tmp_path / "session"
    session.mkdir()

    result = runner.invoke(cli, ["filter", "-s", str(session)])
    assert result.exit_code == 0
    mock_run.assert_called_once()

def test_cli_group_exists(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Subject Frame Extractor CLI." in result.output
