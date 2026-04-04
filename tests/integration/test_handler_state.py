from tests.mock_app import build_mock_app
from ui.handlers.pipeline_handlers import PipelineHandler


def test_handler_is_bound_to_correct_instance(tmp_path):
    """Tier 2: Verify that the PipelineHandler is bound to the specific AppUI instance."""
    # Create a fresh app instance using the factory
    app = build_mock_app(downloads_dir=str(tmp_path))

    # Check that the pipeline_handler exists
    assert hasattr(app, "pipeline_handler")
    assert isinstance(app.pipeline_handler, PipelineHandler)

    # CRITICAL: Verify the handler's 'app' attribute is the SAME object as our app
    # This prevents "Dual Instance" bugs where handlers update the wrong UI.
    assert app.pipeline_handler.app is app

    # Verify config propagation
    assert app.pipeline_handler.config.downloads_dir == str(tmp_path)


def test_multiple_instances_are_isolated(tmp_path):
    """Tier 2: Verify that multiple factory calls return isolated instances."""
    dir1 = tmp_path / "app1"
    dir2 = tmp_path / "app2"
    dir1.mkdir()
    dir2.mkdir()

    app1 = build_mock_app(downloads_dir=str(dir1))
    app2 = build_mock_app(downloads_dir=str(dir2))

    assert app1 is not app2
    assert app1.pipeline_handler.app is app1
    assert app2.pipeline_handler.app is app2

    assert app1.config.downloads_dir != app2.config.downloads_dir
