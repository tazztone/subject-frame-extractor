import hashlib
import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from core.io_utils import (
    _compute_sha256,
    create_frame_map,
    detect_hwaccel,
    download_model,
    is_image_folder,
    list_images,
    sanitize_filename,
    validate_video_file,
)


def test_validate_video_file_not_found():
    with pytest.raises(FileNotFoundError):
        validate_video_file("non_existent_file.mp4")


def test_validate_video_file_empty(tmp_path):
    empty_file = tmp_path / "empty.mp4"
    empty_file.touch()
    with pytest.raises(ValueError, match="empty"):
        validate_video_file(str(empty_file))


@patch("core.io_utils.cv2.VideoCapture")
def test_validate_video_file_invalid(mock_vc, tmp_path):
    video_file = tmp_path / "invalid.mp4"
    video_file.write_text("not a video")
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_vc.return_value = mock_cap

    with pytest.raises(ValueError, match="Could not open video file"):
        validate_video_file(str(video_file))
    mock_cap.release.assert_called_once()


@patch("core.io_utils.cv2.VideoCapture")
def test_validate_video_file_success(mock_vc, tmp_path):
    video_file = tmp_path / "valid.mp4"
    video_file.write_text("fake video data")
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_vc.return_value = mock_cap

    assert validate_video_file(str(video_file)) is True
    mock_cap.release.assert_called_once()


def test_sanitize_filename(mock_config):
    mock_config.utility_max_filename_length = 100  # Ensure it's an int
    assert sanitize_filename("hello/world.jpg", mock_config) == "hello_world.jpg"
    assert sanitize_filename("a" * 50, mock_config) == "a" * 50
    assert sanitize_filename("chars!@#$%^&*().png", mock_config) == "chars__________.png"


def test_is_image_folder(tmp_path):
    assert is_image_folder(tmp_path) is True
    assert is_image_folder(tmp_path / "non_existent") is False
    assert is_image_folder("") is False
    assert is_image_folder(None) is False


def test_list_images(tmp_path, mock_config):
    mock_config.utility_image_extensions = [".jpg", ".png", ".webp"]
    (tmp_path / "img1.jpg").touch()
    (tmp_path / "img2.png").touch()
    (tmp_path / "not_img.txt").touch()
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "img3.webp").touch()

    # Non-recursive
    imgs = list_images(tmp_path, mock_config, recursive=False)
    assert len(imgs) == 2
    assert any(i.name == "img1.jpg" for i in imgs)
    assert any(i.name == "img2.png" for i in imgs)

    # Recursive
    imgs_rec = list_images(tmp_path, mock_config, recursive=True)
    # Ensure we only count files
    imgs_rec = [i for i in imgs_rec if i.is_file()]
    assert len(imgs_rec) == 3
    assert any(i.name == "img3.webp" for i in imgs_rec)


@patch("subprocess.run")
def test_detect_hwaccel_variants(mock_run):
    logger = MagicMock()

    # CUDA
    mock_run.return_value = MagicMock(stdout="cuda\ncuvid\n", returncode=0)
    assert detect_hwaccel(logger) == ("cuda", None)

    # VAAPI
    mock_run.return_value = MagicMock(stdout="vaapi\n", returncode=0)
    assert detect_hwaccel(logger) == ("vaapi", None)

    # VideoToolbox
    mock_run.return_value = MagicMock(stdout="videotoolbox\n", returncode=0)
    assert detect_hwaccel(logger) == ("videotoolbox", None)

    # None
    mock_run.return_value = MagicMock(stdout="none\n", returncode=0)
    assert detect_hwaccel(logger) == (None, None)

    # Error
    mock_run.side_effect = Exception("ffmpeg not found")
    assert detect_hwaccel(logger) == (None, None)
    assert logger.warning.called


def test_compute_sha256(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert _compute_sha256(f) == expected


@patch("urllib.request.urlopen")
def test_download_model_full(mock_urlopen, tmp_path):
    logger = MagicMock()
    error_handler = MagicMock()
    error_handler.with_retry.return_value = lambda x: x

    url = "http://example.com/model.onnx"
    dest = tmp_path / "model.onnx"
    data = b"fake model data" * 100000  # 1.5MB
    sha256 = hashlib.sha256(data).hexdigest()

    # 1. Successful download with SHA256
    mock_resp = MagicMock()
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.read.side_effect = [data, b""]

    def mock_getheader(name, default=None):
        if name == "Content-Length":
            return str(len(data))
        return default

    mock_resp.getheader.side_effect = mock_getheader
    mock_urlopen.return_value = mock_resp

    download_model(url, dest, "test model", logger, error_handler, "UA", expected_sha256=sha256)
    assert dest.exists()
    assert dest.read_bytes() == data

    # 2. Use cached file (verified by SHA256)
    download_model(url, dest, "test model", logger, error_handler, "UA", expected_sha256=sha256)
    # Check that urlopen was not called again for this call
    assert mock_urlopen.call_count == 1

    # 3. Cached file mismatch (re-download)
    dest.write_text("corrupted")
    mock_resp.read.side_effect = [data, b""]
    # Ensure getheader mock is still valid or reset it
    mock_resp.getheader.side_effect = mock_getheader
    download_model(url, dest, "test model", logger, error_handler, "UA", expected_sha256=sha256)
    assert dest.read_bytes() == data
    assert mock_urlopen.call_count == 2

    dest.unlink()
    mock_resp.read.side_effect = [data, b""]
    mock_resp.getheader.side_effect = lambda name, default=None: str(len(data)) if name == "Content-Length" else default
    download_model(url, dest, "test model", logger, error_handler, "UA", token="secret", min_size=0)
    # Verify headers in Request
    args, kwargs = mock_urlopen.call_args
    req = args[0]
    assert req.get_header("Authorization") == "Bearer secret"

    # 4. Content-Length mismatch (too small)
    mock_resp.getheader.side_effect = lambda name, default=None: "100" if name == "Content-Length" else default
    dest.unlink()
    with pytest.raises(RuntimeError, match="Failed to download required model"):
        download_model(url, dest, "test model", logger, error_handler, "UA", min_size=1000)

    # 5. SHA256 mismatch after download
    mock_resp.getheader.side_effect = lambda name, default=None: "1000000" if name == "Content-Length" else default
    mock_resp.read.side_effect = [data, b""]
    with pytest.raises(RuntimeError, match="Failed to download required model"):
        download_model(url, dest, "test model", logger, error_handler, "UA", expected_sha256="wrong_sha")

    # 6. Retry logic test
    # We'll use a real ErrorHandler to test the sleep and retry
    from core.error_handling import ErrorHandler

    real_handler = ErrorHandler(logger, max_attempts=2, backoff_seconds=[0.001])

    # First attempt fails with URLError, second succeeds
    mock_urlopen.side_effect = [urllib.error.URLError("First Fail"), mock_resp]
    mock_resp.read.side_effect = [data, b""]

    download_model(url, dest, "retry test", logger, real_handler, "UA", min_size=0)
    assert mock_urlopen.call_count >= 2
    assert dest.exists()


def test_create_frame_map(tmp_path):
    logger = MagicMock()
    (tmp_path / "frame_map.json").write_text(json.dumps(["10", "20", "5"]))

    fmap = create_frame_map(tmp_path, logger, ext=".png")
    # Sorted frames: 5, 10, 20
    assert fmap == {5: "frame_000001.png", 10: "frame_000002.png", 20: "frame_000003.png"}


def test_create_frame_map_json_error(tmp_path):
    logger = MagicMock()
    # Corrupt JSON
    (tmp_path / "frame_map.json").write_text("{invalid")
    fmap = create_frame_map(tmp_path, logger)
    assert fmap == {}
    assert logger.error.called
