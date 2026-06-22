import json
from unittest.mock import MagicMock, patch

import pytest

from core.managers.media_session import (
    MediaSession,
    load_analysis_scenes,
    validate_dir,
)


def test_validate_session_dir(tmp_path):
    d = tmp_path / "valid"
    d.mkdir()
    path, error = validate_dir(str(d))
    assert path is not None
    assert error is None

    path, error = validate_dir(str(tmp_path / "invalid"))
    assert path is None
    assert error is not None


def test_execute_session_load(mock_logger, tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "run_config.json").write_text(json.dumps({"source_path": "test.mp4"}))
    (session_dir / "scenes.json").write_text(json.dumps([[0, 10]]))

    event = MagicMock()
    event.session_path = str(session_dir)

    res = MediaSession.execute_session_load(event, mock_logger)
    assert res["success"] is True
    assert res["session_path"] == str(session_dir)
    assert len(res["scenes"]) == 1


def test_execute_session_load_errors(mock_logger, tmp_path):
    event = MagicMock()

    # 1. Empty path
    event.session_path = ""
    res = MediaSession.execute_session_load(event, mock_logger)
    assert "error" in res
    assert "Please enter a path" in res["error"]

    # 2. Invalid path
    event.session_path = str(tmp_path / "ghost")
    res = MediaSession.execute_session_load(event, mock_logger)
    assert "error" in res
    assert "Session directory does not exist" in res["error"]

    # 3. Missing run_config.json
    session_dir = tmp_path / "session_err"
    session_dir.mkdir()
    event.session_path = str(session_dir)
    res = MediaSession.execute_session_load(event, mock_logger)
    assert "error" in res
    assert "run_config.json" in res["error"]

    # 4. Invalid run_config.json
    (session_dir / "run_config.json").write_text("invalid json")
    res = MediaSession.execute_session_load(event, mock_logger)
    assert "error" in res
    assert "run_config.json is invalid" in res["error"]

    # 5. Invalid scenes.json (error handled gracefully by skipping scenes)
    (session_dir / "run_config.json").write_text(json.dumps({"source_path": "x"}))
    (session_dir / "scenes.json").write_text("invalid")
    res = MediaSession.execute_session_load(event, mock_logger)
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

    res = MediaSession.execute_session_load(event, mock_logger)
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

    res = MediaSession.execute_session_load(event, mock_logger)
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

    res = MediaSession.execute_session_load(event, mock_logger)
    assert res["success"] is True
    assert len(res["scenes"]) == 1


def test_load_analysis_scenes_folder_mode():
    data = [{"shot_id": 0, "start_frame": 0, "end_frame": 10, "status": "excluded"}]
    # In folder mode, even excluded scenes are loaded regardless of include_only
    scenes = load_analysis_scenes(data, is_folder_mode=True, include_only=False)
    assert len(scenes) == 1

    scenes = load_analysis_scenes(data, is_folder_mode=True, include_only=True)
    assert len(scenes) == 1

    # In video mode (is_folder_mode=False), include_only filters excluded
    scenes = load_analysis_scenes(data, is_folder_mode=False, include_only=True)
    assert len(scenes) == 0


def test_validate_session_dir_exception():
    # Trigger exception in validate_dir (e.g. by passing something that isn't a string or Path and doesn't have expanduser)
    path, error = validate_dir(None)
    assert path is None
    assert "Invalid session path" in error


# --- MediaSession / VideoManager preparation tests ---


def test_media_session_prepare_local(mock_config):
    vm = MediaSession(mock_config, "test.mp4")
    with patch("core.managers.media_session.validate_video_file", return_value="test.mp4") as mock_val:
        path = vm.prepare_video(MagicMock())
        mock_val.assert_called_once()
        assert path == "test.mp4"


@patch("core.managers.media_session.ytdlp.YoutubeDL")
def test_media_session_prepare_youtube(mock_ytdl, mock_config, mock_logger):
    vm = MediaSession(mock_config, "https://youtube.com/watch?v=123")

    mock_instance = mock_ytdl.return_value.__enter__.return_value
    mock_instance.extract_info.return_value = {}
    mock_instance.prepare_filename.return_value = "downloaded.mp4"

    with patch("core.managers.media_session.validate_video_file", return_value="downloaded.mp4"):
        path = vm.prepare_video(mock_logger)

    assert path == "downloaded.mp4"
    mock_instance.extract_info.assert_called_once()


def test_media_session_invalid_inputs(mock_config, mock_logger):
    # Invalid URL/File
    vm = MediaSession(mock_config, "invalid_file.mp4")
    # Assuming validate_video_file raises FileNotFoundError
    with patch("core.managers.media_session.validate_video_file", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            vm.prepare_video(mock_logger)


@patch("core.managers.media_session.DownloadError", new_callable=lambda: type("DownloadError", (Exception,), {}))
@patch("core.managers.media_session.ytdlp")
def test_media_session_youtube_error(mock_ytdlp_module, mock_download_error_cls, mock_config, mock_logger):
    # Setup the YoutubeDL context manager mock
    mock_ctx = mock_ytdlp_module.YoutubeDL.return_value.__enter__.return_value
    mock_ctx.extract_info.side_effect = mock_download_error_cls("Failed")

    vm = MediaSession(mock_config, "https://youtube.com/watch?v=bad")

    with pytest.raises(RuntimeError) as excinfo:
        vm.prepare_video(mock_logger)

    assert "Download failed" in str(excinfo.value)


@patch("cv2.VideoCapture")
def test_get_video_info(mock_cap):
    instance = mock_cap.return_value
    instance.isOpened.return_value = True

    import cv2

    def get_side_effect(prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 1920
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 1080
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return 100
        return 0

    instance.get.side_effect = get_side_effect

    info = MediaSession.get_video_info("test.mp4")

    assert info["fps"] == 30.0
    assert info["width"] == 1920
    assert info["height"] == 1080
    assert info["frame_count"] == 100
