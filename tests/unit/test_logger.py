import logging
from queue import Queue
from unittest.mock import MagicMock, patch

from core.logger import AppLogger, ColoredFormatter, GradioQueueHandler, LogEvent, setup_logging


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
