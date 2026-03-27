import json
import logging
from unittest.mock import MagicMock

from core.io_utils import create_frame_map
from core.logger import AppLogger, log_with_component


def test_logger_interop(tmp_path):
    # Setup
    config = MagicMock()
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    config.logs_dir = str(logs_dir)
    config.log_format = "%(message)s"
    config.log_colored = False
    config.log_level = "INFO"
    config.log_structured_path = "structured.json"

    app_logger = AppLogger(config)
    std_logger = logging.getLogger("test_std")
    std_logger.setLevel(logging.INFO)

    # Test log_with_component with AppLogger
    with MagicMock() as mock_log:
        app_logger._log = mock_log
        log_with_component(app_logger, "info", "test message", component="test_comp")
        mock_log.assert_called_once_with("INFO", "test message", "test_comp")

    # Test log_with_component with standard logger
    with MagicMock() as mock_info:
        std_logger.info = mock_info
        log_with_component(std_logger, "info", "test message", component="test_comp")
        # Should call info with extra={'component': 'test_comp'}
        mock_info.assert_called_once()
        args, kwargs = mock_info.call_args
        assert args[0] == "test message"
        assert kwargs["extra"]["component"] == "test_comp"


def test_create_frame_map_interop(tmp_path):
    # Setup
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    frame_map_path = output_dir / "frame_map.json"
    with open(frame_map_path, "w") as f:
        json.dump([1, 2, 3], f)

    std_logger = logging.getLogger("test_std_map")

    # This should not raise TypeError anymore
    res = create_frame_map(output_dir, std_logger)
    assert len(res) == 3
    assert res[1] == "frame_000001.webp"


def test_setup_logging_stable_name(tmp_path):
    from core.logger import setup_logging

    config = MagicMock()
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    config.logs_dir = str(logs_dir)
    config.log_format = "%(message)s"
    config.log_colored = False
    config.log_level = "INFO"
    config.log_structured_path = "structured.json"

    log_dir = tmp_path / "custom_logs"

    # Test stable_log_name=True
    session_log = setup_logging(config, log_dir=log_dir, stable_log_name=True)
    assert session_log.name == "run.log"
    assert session_log.parent == log_dir
    assert session_log.exists()

    # Test stable_log_name=False (default)
    session_log_2 = setup_logging(config, log_dir=log_dir, stable_log_name=False)
    assert session_log_2.name.startswith("session_")
    assert session_log_2.name.endswith(".log")
    assert session_log_2.exists()
