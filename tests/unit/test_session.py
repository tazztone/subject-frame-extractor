import json
from unittest.mock import MagicMock

import pytest

from core.managers.session import _load_analysis_scenes, execute_session_load, validate_session_dir
from core.models import Scene


@pytest.fixture
def mock_logger():
    return MagicMock()


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


def test_load_analysis_scenes():
    data = [{"shot_id": 0, "start_frame": 0, "end_frame": 10, "status": "included"}]
    scenes = _load_analysis_scenes(data, is_folder_mode=False)
    assert len(scenes) == 1
    assert isinstance(scenes[0], Scene)

    data[0]["status"] = "excluded"
    scenes = _load_analysis_scenes(data, is_folder_mode=False)
    assert len(scenes) == 0
