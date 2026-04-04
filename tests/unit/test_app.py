import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in path
project_root = str(Path(__file__).parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_parse_args():
    import app

    with patch("sys.argv", ["app.py", "--server-name", "0.0.0.0", "--server-port", "7860"]):
        args = app.parse_args()
        assert args.server_name == "0.0.0.0"
        assert args.server_port == 7860


def test_cleanup_models():
    import app

    mock_registry = MagicMock()
    with (
        patch("app.torch.cuda.is_available", return_value=True, create=True),
        patch("app.torch.cuda.empty_cache") as mock_empty_cache,
        patch("app.gc.collect") as mock_gc,
    ):
        app.cleanup_models(mock_registry)
        mock_registry.clear.assert_called_once()
        mock_empty_cache.assert_called_once()
        mock_gc.assert_called_once()


def test_main_success():
    import app

    with (
        patch("app.AppUI") as mock_ui,
        patch("app.setup_logging"),
        patch("app.AppLogger"),
        patch("app.ModelRegistry"),
        patch("app.ThumbnailManager"),
        patch("app.Config") as mock_config,
    ):
        # Setup mocks
        mock_config_instance = mock_config.return_value
        mock_config_instance.server_name = "127.0.0.1"
        mock_config_instance.server_port = 7860
        mock_config_instance.share = False
        mock_config_instance.ssl_keyfile = None
        mock_config_instance.ssl_certfile = None
        mock_config_instance.ssl_verify = None
        mock_config_instance.auth = None
        mock_config_instance.downloads_dir = "downloads"
        mock_config_instance.allowed_paths = []

        mock_ui_instance = mock_ui.return_value
        mock_demo = mock_ui_instance.build_ui.return_value

        with patch("sys.argv", ["app.py"]):
            app.main()

        mock_ui_instance.build_ui.assert_called_once()
        mock_demo.launch.assert_called_once()


def test_main_keyboard_interrupt():
    import app

    with (
        patch("app.AppUI") as mock_ui,
        patch("app.setup_logging"),
        patch("app.AppLogger") as mock_logger,
    ):
        mock_ui.side_effect = KeyboardInterrupt()
        with patch("sys.argv", ["app.py"]):
            # Should not raise exception
            app.main()
        mock_logger.return_value.info.assert_any_call("\nApplication stopped by user")


def test_main_exception():
    import app

    with (
        patch("app.AppUI") as mock_ui,
        patch("app.setup_logging"),
        patch("app.AppLogger") as mock_logger,
        patch("sys.exit") as mock_exit,
    ):
        mock_ui.side_effect = Exception("Test Error")
        with patch("sys.argv", ["app.py"]):
            app.main()
        mock_exit.assert_called_once_with(1)
        mock_logger.return_value.error.assert_called()
