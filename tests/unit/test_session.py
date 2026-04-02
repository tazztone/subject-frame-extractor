import json
from unittest.mock import MagicMock

from core.managers.session import _load_analysis_scenes, execute_session_load, validate_session_dir


def test_validate_session_dir(tmp_path):
    d = tmp_path / "valid"
    d.mkdir()
    path, error = validate_session_dir(str(d))
    assert path is not None
    assert error is None

    path, error = validate_session_dir(str(tmp_path / "invalid"))
    assert path is None
    assert error is not None


def test_execute_session_load(mock_logger, tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "run_config.json").write_text(json.dumps({"source_path": "test.mp4"}))
    (session_dir / "scenes.json").write_text(json.dumps([[0, 10]]))

    event = MagicMock()
    event.session_path = str(session_dir)

    res = execute_session_load(event, mock_logger)
    assert res["success"] is True
    assert res["session_path"] == str(session_dir)
    assert len(res["scenes"]) == 1


def test_execute_session_load_errors(mock_logger, tmp_path):
    event = MagicMock()

    # 1. Empty path
    event.session_path = ""
    res = execute_session_load(event, mock_logger)
    assert "error" in res
    assert "Please enter a path" in res["error"]

    # 2. Invalid path
    event.session_path = str(tmp_path / "ghost")
    res = execute_session_load(event, mock_logger)
    assert "error" in res
    assert "Session directory does not exist" in res["error"]

    # 3. Missing run_config.json
    session_dir = tmp_path / "session_err"
    session_dir.mkdir()
    event.session_path = str(session_dir)
    res = execute_session_load(event, mock_logger)
    assert "error" in res
    assert "run_config.json" in res["error"]

    # 4. Invalid run_config.json
    (session_dir / "run_config.json").write_text("invalid json")
    res = execute_session_load(event, mock_logger)
    assert "error" in res
    assert "run_config.json is invalid" in res["error"]

    # 5. Invalid scenes.json (error handled gracefully by skipping scenes)
    (session_dir / "run_config.json").write_text(json.dumps({"source_path": "x"}))
    (session_dir / "scenes.json").write_text("invalid")
    res = execute_session_load(event, mock_logger)
    assert "error" in res
    assert "Failed to read scenes.json" in res["error"]


def test_execute_session_load_with_seeds(mock_logger, tmp_path):
    session_dir = tmp_path / "session_seeds"
    session_dir.mkdir()
    (session_dir / "run_config.json").write_text(json.dumps({"source_path": "test.mp4"}))
    (session_dir / "scenes.json").write_text(json.dumps([[0, 10]]))
    (session_dir / "scene_seeds.json").write_text(json.dumps({"0": {"best_frame": 5, "status": "included"}}))

    event = MagicMock()
    event.session_path = str(session_dir)

    res = execute_session_load(event, mock_logger)
    assert res["success"] is True
    assert res["scenes"][0]["best_frame"] == 5


def test_execute_session_load_corrupt_seeds(mock_logger, tmp_path):
    session_dir = tmp_path / "session_corrupt_seeds"
    session_dir.mkdir()
    (session_dir / "run_config.json").write_text(json.dumps({"source_path": "test.mp4"}))
    (session_dir / "scenes.json").write_text(json.dumps([[0, 10]]))
    # Corrupt JSON for scene_seeds
    (session_dir / "scene_seeds.json").write_text("{invalid")

    event = MagicMock()
    event.session_path = str(session_dir)

    res = execute_session_load(event, mock_logger)
    # Should still succeed but log warning about seeds
    assert res["success"] is True
    assert len(res["scenes"]) == 1
    mock_logger.warning.assert_called()
    assert "Could not load scene_seeds.json" in mock_logger.warning.call_args[0][0]


def test_execute_session_load_missing_seeds_file(mock_logger, tmp_path):
    session_dir = tmp_path / "session_no_seeds"
    session_dir.mkdir()
    (session_dir / "run_config.json").write_text(json.dumps({"source_path": "test.mp4"}))
    (session_dir / "scenes.json").write_text(json.dumps([[0, 10]]))
    # No scene_seeds.json file

    event = MagicMock()
    event.session_path = str(session_dir)

    res = execute_session_load(event, mock_logger)
    assert res["success"] is True
    assert len(res["scenes"]) == 1


def test_load_analysis_scenes_folder_mode():
    data = [{"shot_id": 0, "start_frame": 0, "end_frame": 10, "status": "excluded"}]
    # In folder mode, even excluded scenes are loaded regardless of include_only
    scenes = _load_analysis_scenes(data, is_folder_mode=True, include_only=False)
    assert len(scenes) == 1

    scenes = _load_analysis_scenes(data, is_folder_mode=True, include_only=True)
    assert len(scenes) == 1

    # In video mode (is_folder_mode=False), include_only filters excluded
    scenes = _load_analysis_scenes(data, is_folder_mode=False, include_only=True)
    assert len(scenes) == 0


def test_validate_session_dir_exception():
    # Trigger exception in validate_session_dir (e.g. by passing something that isn't a string or Path and doesn't have expanduser)
    path, error = validate_session_dir(None)
    assert path is None
    assert "Invalid session path" in error
