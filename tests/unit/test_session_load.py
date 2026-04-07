from unittest.mock import patch

from core.application_state import ApplicationState
from tests.mock_app import build_mock_app


@patch("ui.handlers.pipeline_handlers.execute_session_load")
def test_session_load_restores_state(mock_load, tmp_path):
    """Verify run_session_load_wrapper correctly restores UI and state components."""
    app = build_mock_app(downloads_dir=str(tmp_path))

    # Mock execute_session_load return value
    mock_run_config = {
        "video_path": "test_video.mp4",
        "source_path": "input.mp4",
        "max_resolution": "1080",
        "output_folder": "outputs",
    }

    # Mock the return from execute_session_load
    mock_result = {"run_config": mock_run_config, "session_path": str(tmp_path), "scenes": [], "metadata_exists": False}
    mock_load.return_value = mock_result

    # Trigger the load handler
    # Note: We need to pass ApplicationState as it's the standard wrapper signature
    current_state = ApplicationState()

    # We need to call the wrapper and drain the generator
    updates_generator = app.pipeline_handler.run_session_load_wrapper(str(tmp_path), current_state)

    results = list(updates_generator)
    print("====== RESULTS ======")
    for r in results:
        print(r)
    print("=====================")

    # Verify status updates and final state
    assert any("Session Loaded" in str(u.get(app.components["unified_status"], "")) for u in results)

    # Check final update (the dict with all restorations)
    final_update = results[-1]

    # Verify application_state restoration
    restored_state = final_update[app.components["application_state"]]
    assert restored_state.extracted_video_path == "test_video.mp4"

    # Verify component value updates
    assert final_update[app.components["source_input"]].get("value") == "input.mp4"
