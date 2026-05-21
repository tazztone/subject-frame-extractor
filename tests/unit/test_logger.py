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


def test_app_logger_success():
    """Test explicit AppLogger.success method."""
    config = MagicMock()
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = mock_get_logger.return_value
        logger = AppLogger(config)

        logger.success("Success message", component="test_comp")

        mock_logger.log.assert_called_with(
            25, "Success message [test_comp]", extra={"component": "test_comp"}, exc_info=None
        )


def test_app_logger_log():
    """Test AppLogger.log mapping."""
    config = MagicMock()
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = mock_get_logger.return_value
        logger = AppLogger(config)

        logger.log(logging.ERROR, "Error msg", component="err_comp", custom="field")

        mock_logger.log.assert_called_with(
            logging.ERROR, "Error msg [err_comp]", extra={"component": "err_comp", "custom": "field"}, exc_info=None
        )


from core.logger import log_with_component


def test_log_with_component_none():
    """Test log_with_component with None logger."""
    log_with_component(None, "INFO", "msg")


def test_log_with_component_app_logger():
    """Test log_with_component with AppLogger."""
    config = MagicMock()
    logger = AppLogger(config)

    with patch.object(logger, "info") as mock_info:
        log_with_component(logger, "INFO", "My info msg", component="my_comp", custom="val")
        mock_info.assert_called_with("My info msg", component="my_comp", custom="val")


def test_log_with_component_std_logger():
    """Test log_with_component with standard logger."""
    logger = logging.getLogger("test_logger")

    with patch.object(logger, "error") as mock_error:
        log_with_component(logger, "ERROR", "My err msg", component="std_comp", custom="val")
        mock_error.assert_called_with("My err msg", extra={"component": "std_comp"}, custom="val")


def test_log_with_component_std_logger_success():
    """Test log_with_component with standard logger and custom SUCCESS level fallback."""
    logger = logging.getLogger("test_logger")

    with patch.object(logger, "log") as mock_log:
        log_with_component(logger, "SUCCESS", "My success msg", component="std_comp")
        mock_log.assert_called_with(25, "My success msg [std_comp]")


def test_log_with_component_missing_method():
    """Test log_with_component when level method doesn't exist on logger."""
    logger = logging.getLogger("test_logger")
    log_with_component(logger, "NONEXISTENT", "msg") # Should not crash


def test_app_logger_copy_log_to_output(tmp_path):
    """Test copy_log_to_output method."""
    config = MagicMock()
    session_log = tmp_path / "test_session.log"
    session_log.write_text("log content")

    logger = AppLogger(config, session_log_file=session_log)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    logger.copy_log_to_output(output_dir)

    copied_log = output_dir / "session.log"
    assert copied_log.exists()
    assert copied_log.read_text() == "log content"

def test_app_logger_copy_log_to_output_exception(tmp_path):
    """Test copy_log_to_output handles exceptions."""
    config = MagicMock()
    session_log = tmp_path / "test_session.log"
    session_log.write_text("log content")

    logger = AppLogger(config, session_log_file=session_log)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with patch("shutil.copy", side_effect=Exception("Copy failed")):
        logger.copy_log_to_output(output_dir)
        # Should catch and pass silently
        assert not (output_dir / "session.log").exists()

def test_app_logger_copy_log_to_output_no_log_file():
    """Test copy_log_to_output method when session_log_file is None."""
    config = MagicMock()
    logger = AppLogger(config) # no session_log_file passed

    with patch("shutil.copy") as mock_copy:
        logger.copy_log_to_output("some_dir")
        mock_copy.assert_not_called()

def test_app_logger_copy_log_to_output_file_not_exists(tmp_path):
    """Test copy_log_to_output method when session_log_file does not exist."""
    config = MagicMock()
    session_log = tmp_path / "test_session.log"
    logger = AppLogger(config, session_log_file=session_log)

    with patch("shutil.copy") as mock_copy:
        logger.copy_log_to_output("some_dir")
        mock_copy.assert_not_called()

def test_app_logger_level_methods():
    """Test all specific AppLogger level methods explicitly."""
    config = MagicMock()
    with patch("logging.getLogger") as mock_get_logger:
        mock_logger = mock_get_logger.return_value
        logger = AppLogger(config)

        logger.debug("Debug msg", component="comp1")
        mock_logger.log.assert_called_with(logging.DEBUG, "Debug msg [comp1]", extra={"component": "comp1"}, exc_info=None)

        logger.info("Info msg", component="comp2")
        mock_logger.log.assert_called_with(logging.INFO, "Info msg [comp2]", extra={"component": "comp2"}, exc_info=None)

        logger.warning("Warn msg", component="comp3")
        mock_logger.log.assert_called_with(logging.WARNING, "Warn msg [comp3]", extra={"component": "comp3"}, exc_info=None)

        logger.error("Err msg", component="comp4")
        mock_logger.log.assert_called_with(logging.ERROR, "Err msg [comp4]", extra={"component": "comp4"}, exc_info=None)

        logger.critical("Crit msg", component="comp5")
        mock_logger.log.assert_called_with(logging.CRITICAL, "Crit msg [comp5]", extra={"component": "comp5"}, exc_info=None)
