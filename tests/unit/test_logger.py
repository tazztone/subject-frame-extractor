import json
import logging
import sys
from queue import Queue
from unittest.mock import MagicMock, patch

from core.logger import (
    SUCCESS_LEVEL_NUM,
    AppLogger,
    ColoredFormatter,
    GradioQueueHandler,
    JSONFormatter,
    LogEvent,
    log_with_component,
    setup_logging,
)


def test_log_event_model():
    """Test LogEvent Pydantic model."""
    event = LogEvent(timestamp="2023-01-01T00:00:00", level="INFO", message="test", component="system")
    assert event.message == "test"
    assert event.level == "INFO"


def test_gradio_queue_handler():
    """Test GradioQueueHandler emits to queue."""
    queue = Queue()
    handler = GradioQueueHandler(queue)
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0, msg="hello", args=(), exc_info=None
    )
    # Mock formatter
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.emit(record)

    assert not queue.empty()
    item = queue.get()
    assert item["log"] == "hello"


def test_colored_formatter():
    """Test ColoredFormatter adds color codes."""
    formatter = ColoredFormatter("%(levelname)s: %(message)s")
    record = logging.LogRecord(
        name="test", level=logging.ERROR, pathname="", lineno=0, msg="error msg", args=(), exc_info=None
    )
    formatted = formatter.format(record)
    # Should contain ANSI color code for ERROR ( \033[31m )
    assert "\033[31mERROR\033[0m" in formatted


@patch("logging.config.dictConfig")
def test_setup_logging(mock_dict_config, tmp_path):
    """Test setup_logging configuration logic."""
    config = MagicMock()
    config.logs_dir = str(tmp_path)
    config.log_format = "%(message)s"
    config.log_colored = True
    config.log_level = "INFO"

    log_file = setup_logging(config)

    assert mock_dict_config.called
    assert log_file.parent == tmp_path
    assert "session_" in log_file.name


@patch("logging.config.dictConfig")
def test_setup_logging_stable_name(mock_dict_config, tmp_path):
    """Test setup_logging configuration logic with stable name."""
    config = MagicMock()
    config.logs_dir = str(tmp_path)
    config.log_format = "%(message)s"
    config.log_colored = True
    config.log_level = "INFO"

    log_file = setup_logging(config, stable_log_name=True)

    assert mock_dict_config.called
    assert log_file.parent == tmp_path
    assert log_file.name == "run.log"


@patch("logging.config.dictConfig")
def test_setup_logging_with_queue(mock_dict_config, tmp_path):
    """Test setup_logging with a progress queue."""
    config = MagicMock()
    config.logs_dir = str(tmp_path)
    config.log_format = "%(message)s"
    config.log_colored = False
    config.log_level = "INFO"
    queue = Queue()

    setup_logging(config, progress_queue=queue)
    assert mock_dict_config.called
    args, kwargs = mock_dict_config.call_args
    conf = args[0]
    assert "gradio" in conf["handlers"]
    assert "gradio" in conf["loggers"]["app_logger"]["handlers"]


def test_app_logger_all_methods():
    """Test all AppLogger proxy methods."""
    config = MagicMock()
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = mock_get_logger.return_value
        logger = AppLogger(config)

        methods = ["debug", "info", "warning", "error", "success", "critical"]
        for method_name in methods:
            method = getattr(logger, method_name)
            method(f"{method_name} message", component="test")

            level = getattr(logging, method_name.upper(), logging.INFO)
            if method_name == "success":
                level = 25

            mock_logger.log.assert_any_call(
                level, f"{method_name} message [test]", extra={"component": "test"}, exc_info=None
            )


def test_app_logger_critical():
    """Test AppLogger.critical specifically calls _log with correct args."""
    config = MagicMock()
    logger = AppLogger(config)
    with patch.object(logger, "_log") as mock_log:
        logger.critical("Critical error occurred", component="test_component", exc_info=True)
        mock_log.assert_called_once_with("CRITICAL", "Critical error occurred", "test_component", exc_info=True)


@patch("logging.config.dictConfig")
def test_setup_logging_no_console(mock_dict_config, tmp_path):
    """Test setup_logging with console logging disabled."""
    config = MagicMock()
    config.logs_dir = str(tmp_path)
    config.log_format = "%(message)s"
    config.log_colored = False
    config.log_level = "INFO"

    setup_logging(config, log_to_console=False)
    assert mock_dict_config.called
    conf = mock_dict_config.call_args[0][0]
    assert conf["handlers"]["console"]["level"] == "CRITICAL"


@patch("core.logger.Path.mkdir")
@patch("logging.config.dictConfig")
def test_setup_logging_mkdir(mock_dict_config, mock_mkdir, tmp_path):
    """Test setup_logging calls mkdir."""
    config = MagicMock()
    config.logs_dir = str(tmp_path)
    setup_logging(config)
    assert mock_mkdir.called


def test_gradio_queue_handler_error():
    """Test GradioQueueHandler error handling."""
    queue = MagicMock()
    queue.put.side_effect = Exception("Queue full")
    handler = GradioQueueHandler(queue)
    record = MagicMock()

    with patch.object(handler, "handleError") as mock_handle_error:
        handler.emit(record)
        assert mock_handle_error.called


def test_log_with_component_none():
    """Test log_with_component with None logger."""
    # Should just return without crashing
    log_with_component(None, "info", "test message")


def test_log_with_component_none_logger():
    """Test log_with_component ignores None logger."""
    # Should not raise exception
    log_with_component(None, "INFO", "message")


def test_log_with_component_app_logger():
    """Test log_with_component with AppLogger."""
    logger = MagicMock(spec=AppLogger)
    logger.info = MagicMock()
    log_with_component(logger, "INFO", "message", component="test", extra_arg="val")
    logger.info.assert_called_once_with("message", component="test", extra_arg="val")


def test_log_with_component_app_logger_lowercase():
    """Test log_with_component with AppLogger and lowercase level."""
    logger = MagicMock(spec=AppLogger)
    log_with_component(logger, "info", "test message", component="my_component")
    logger.info.assert_called_once_with("test message", component="my_component")


def test_log_with_component_standard_logger():
    """Test log_with_component with standard logging.Logger."""
    logger = logging.getLogger("test_standard")
    with patch.object(logger, "info") as mock_info:
        log_with_component(logger, "INFO", "message", component="test", extra_arg="val")
        mock_info.assert_called_once_with("message", extra={"component": "test"}, extra_arg="val")


def test_log_with_component_standard_logger_mock():
    """Test log_with_component with standard logging.Logger mock."""
    logger = MagicMock(spec=logging.Logger)
    log_with_component(logger, "info", "test message", component="my_component")
    logger.info.assert_called_once_with("test message", extra={"component": "my_component"})


def test_log_with_component_standard_logger_extra_kwargs():
    """Test log_with_component with standard logger and extra kwargs."""
    logger = MagicMock(spec=logging.Logger)
    log_with_component(logger, "info", "test message", component="my_component", extra={"other": "value"})
    logger.info.assert_called_once_with("test message", extra={"other": "value", "component": "my_component"})


def test_log_with_component_success_fallback():
    """Test log_with_component success fallback."""
    logger = logging.getLogger("test_success")
    with patch.object(logger, "log") as mock_log:
        log_with_component(logger, "SUCCESS", "message", component="test", extra_arg="val")
        mock_log.assert_called_once_with(25, "message [test]", extra_arg="val")


def test_log_with_component_success_fallback_mock():
    """Test log_with_component success fallback on standard logger mock."""
    logger = MagicMock(spec=logging.Logger)
    # Ensure success method doesn't exist to trigger fallback
    del logger.success
    log_with_component(logger, "success", "test message", component="my_component")
    logger.log.assert_called_once_with(SUCCESS_LEVEL_NUM, "test message [my_component]")


def test_log_with_component_missing_level():
    """Test log_with_component with a level that doesn't exist."""
    logger = MagicMock(spec=logging.Logger)
    # Should return silently without crashing
    log_with_component(logger, "nonexistent", "test message", component="my_component")


def test_json_formatter():
    """Test JSONFormatter formats log records correctly."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test_logger",
        level=logging.ERROR,
        pathname="path.py",
        lineno=10,
        msg="test message",
        args=(),
        exc_info=(ValueError, ValueError("error"), None),
    )
    # Add extra attribute
    record.__dict__["component"] = "test_component"
    record.__dict__["custom_val"] = {"key": "val"}

    formatted = formatter.format(record)
    assert "test message" in formatted
    assert "ERROR" in formatted
    assert "test_component" in formatted
    assert "ValueError" in formatted
    assert "custom_val" in formatted
    assert "val" in formatted


def test_app_logger_copy_log_to_output(tmp_path):
    """Test copy_log_to_output copies the log file correctly."""
    config = MagicMock()
    session_log_file = tmp_path / "source.log"
    session_log_file.write_text("test log content")

    logger = AppLogger(config, session_log_file=session_log_file)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    logger.copy_log_to_output(output_dir)

    copied_file = output_dir / "session.log"
    assert copied_file.exists()
    assert copied_file.read_text() == "test log content"


def test_app_logger_copy_log_to_output_missing_file(tmp_path):
    """Test copy_log_to_output handles missing source file gracefully."""
    config = MagicMock()
    session_log_file = tmp_path / "missing.log"

    logger = AppLogger(config, session_log_file=session_log_file)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Should not raise an exception
    logger.copy_log_to_output(output_dir)

    copied_file = output_dir / "session.log"
    assert not copied_file.exists()


def test_app_logger_copy_log_to_output_no_session_file(tmp_path):
    """Test copy_log_to_output handles case where session_log_file is not set."""
    config = MagicMock()

    logger = AppLogger(config)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # Should not raise an exception
    logger.copy_log_to_output(output_dir)

    copied_file = output_dir / "session.log"
    assert not copied_file.exists()


def test_app_logger_copy_log_to_output_exception(tmp_path):
    """Test copy_log_to_output catches and ignores exceptions."""
    config = MagicMock()
    session_log_file = tmp_path / "source.log"
    session_log_file.write_text("test log content")

    logger = AppLogger(config, session_log_file=session_log_file)
    output_dir = tmp_path / "output"

    # Passing an invalid path or missing directory to trigger shutil.copy exception
    # It should not raise since it has a blanket except Exception pass
    with patch("shutil.copy", side_effect=Exception("Disk error")):
        logger.copy_log_to_output(output_dir)


def test_app_logger_log():
    """Test standard log method mapping to internal _log."""
    config = MagicMock()
    logger = AppLogger(config)

    with patch.object(logger, "_log") as mock_log:
        logger.log(logging.ERROR, "An error log", component="custom")
        mock_log.assert_called_once_with("ERROR", "An error log", "custom")


def test_json_formatter_basic():
    """Test JSONFormatter formats basic log record correctly."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="test message",
        args=(),
        exc_info=None,
    )
    record.__dict__["component"] = "test_component"
    record.__dict__["custom_key"] = "custom_value"

    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert data["message"] == "test message"
    assert data["level"] == "INFO"
    assert data["component"] == "test_component"
    assert data["custom_fields"]["custom_key"] == "custom_value"
    assert "timestamp" in data
    assert data["error_type"] is None


def test_json_formatter_sanitize():
    """Test JSONFormatter _sanitize function for various types."""
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.DEBUG,
        pathname="",
        lineno=0,
        msg="test sanitize",
        args=(),
        exc_info=None,
    )

    record.__dict__["a_set"] = {3, 1, 2}
    record.__dict__["a_dict"] = {"inner": {2, 1}, "num": 1.5}
    record.__dict__["a_list"] = [1, {3, 2}, "str"]
    record.__dict__["an_object"] = object()

    formatted = formatter.format(record)
    data = json.loads(formatted)

    fields = data["custom_fields"]
    assert fields["a_set"] == [1, 2, 3]  # sorted elements
    assert fields["a_dict"]["inner"] == [1, 2]  # sorted elements
    assert fields["a_dict"]["num"] == 1.5
    assert fields["a_list"] == [1, [2, 3], "str"]
    assert "object at" in fields["an_object"]  # str(object())


def test_json_formatter_exception():
    """Test JSONFormatter formats exception info properly."""
    formatter = JSONFormatter()

    try:
        raise ValueError("test error")
    except ValueError:
        exc_info = sys.exc_info()

    record = logging.LogRecord(
        name="test",
        level=logging.ERROR,
        pathname="",
        lineno=0,
        msg="error occurred",
        args=(),
        exc_info=exc_info,
    )

    formatted = formatter.format(record)
    data = json.loads(formatted)

    assert data["message"] == "error occurred"
    assert data["level"] == "ERROR"
    assert data["error_type"] == "ValueError"
    assert data["stack_trace"] is not None
    assert "test error" in data["stack_trace"]
