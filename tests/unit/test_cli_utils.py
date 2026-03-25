import threading
from queue import Queue
from unittest.mock import MagicMock, patch

import click
import pytest

from core.cli_utils import _run_pipeline, _setup_runtime


def test_setup_runtime(tmp_path):
    output_dir = tmp_path / "logs"
    output_dir.mkdir()

    with patch("core.cli_utils.setup_logging") as mock_setup:
        config, logger, progress_queue, cancel_event, registry, tm = _setup_runtime(output_dir, verbose=True)

        assert config.log_level == "DEBUG"
        assert not config.monitoring_memory_watchdog_enabled
        assert isinstance(progress_queue, Queue)
        assert isinstance(cancel_event, threading.Event)
        assert mock_setup.called


def test_run_pipeline_success():
    def mock_gen():
        yield {"unified_log": "Stage 1", "done": False}
        yield {"unified_log": "Done", "done": True}

    with patch("click.echo") as mock_echo, patch("click.secho") as mock_secho:
        res = _run_pipeline(mock_gen(), "TestStage")
        assert res["done"] is True
        mock_echo.assert_any_call("  Stage 1")
        mock_secho.assert_any_call("✓ TestStage complete.", fg="green")


def test_run_pipeline_error_in_gen():
    def mock_gen():
        yield {"unified_log": "Stage 1", "done": False}
        raise RuntimeError("Something went wrong")

    with patch("click.secho") as mock_secho:
        with pytest.raises(click.ClickException) as excinfo:
            _run_pipeline(mock_gen(), "TestStage")
        assert "TestStage aborted" in str(excinfo.value)
        mock_secho.assert_called_with("Critial error in TestStage: Something went wrong", fg="red")


def test_run_pipeline_failed_result():
    def mock_gen():
        yield {"unified_log": "Failed", "done": False}

    with pytest.raises(click.ClickException) as excinfo:
        _run_pipeline(mock_gen(), "TestStage")
    assert "TestStage failed: Failed" in str(excinfo.value)


def test_run_pipeline_no_result():
    def mock_gen():
        if False:
            yield

    with pytest.raises(click.ClickException) as excinfo:
        _run_pipeline(mock_gen(), "TestStage")
    assert "TestStage failed unexpectedly" in str(excinfo.value)


def test_run_pipeline_close_generator():
    mock_gen_obj = MagicMock()
    mock_gen_obj.__iter__.return_value = [{"unified_log": "Done", "done": True}]

    _run_pipeline(mock_gen_obj, "TestStage")
    assert mock_gen_obj.close.called
