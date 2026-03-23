import json
import subprocess
import threading
from queue import Queue
from unittest.mock import MagicMock, patch

import pytest

from core.managers.extraction import (
    ExtractionPipeline,
    _process_ffmpeg_showinfo,
    _process_ffmpeg_stream,
    run_ffmpeg_extraction,
)
from core.models import AnalysisParameters


@pytest.fixture
def mock_deps():
    config = MagicMock()
    config.downloads_dir = "/tmp/downloads"
    config.ffmpeg_hwaccel = "off"
    config.ffmpeg_thumbnail_quality = 80
    config.retry_max_attempts = 1
    config.retry_backoff_seconds = 0
    return {
        "config": config,
        "logger": MagicMock(),
        "progress_queue": Queue(),
        "cancel_event": threading.Event(),
    }


@pytest.fixture
def analysis_params():
    params = AnalysisParameters(
        source_path="video.mp4",
        output_folder="/tmp/output",
        method="every_nth_frame",
        nth_frame=10,
        thumbnails_only=True,
    )
    return params


def test_process_ffmpeg_stream(mock_deps):
    tracker = MagicMock()
    tracker.total = 100

    # Mock stream with frame progress
    stream = MagicMock()
    stream.readline.side_effect = ["frame=10\n", "out_time_us=5000000\n", "progress=end\n", ""]

    _process_ffmpeg_stream(stream, tracker, "Test", total_duration_s=10, start_time_s=0)

    assert tracker.set.called
    # 5s / 10s = 0.5, 0.5 * 100 = 50
    tracker.set.assert_any_call(50, desc="Test")


def test_process_ffmpeg_showinfo():
    stream = MagicMock()
    stream.readline.side_effect = [
        "[Parsed_showinfo_0 @ 0x...] n:0 pts:0 pts_time:0.0 pos:0 ...\n",
        "[Parsed_showinfo_0 @ 0x...] n:1 pts:3003 pts_time:0.1001 pos:4096 ...\n",
        "",
    ]

    frame_numbers, full_stderr = _process_ffmpeg_showinfo(stream, fps=30.0)

    assert frame_numbers == [0, 3]  # 0.1001 * 30 = 3.003 -> 3
    assert "pts_time:0.1001" in full_stderr


@patch("core.managers.extraction.subprocess.Popen")
@patch("core.managers.extraction.detect_hwaccel", create=True)
def test_run_ffmpeg_extraction_success(mock_detect, mock_popen_cls, mock_deps, analysis_params, tmp_path):
    mock_detect.return_value = (None, None)
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    video_info = {"width": 1920, "height": 1080, "fps": 30, "frame_count": 300}

    # Mock process
    mock_process = mock_popen_cls.return_value
    mock_process.stdout = MagicMock()
    mock_process.stdout.readline.return_value = ""
    mock_process.stderr = MagicMock()
    mock_process.stderr.readline.return_value = ""
    mock_process.poll.return_value = 0

    with patch("core.managers.extraction.subprocess.run"):
        run_ffmpeg_extraction(
            "video.mp4",
            output_dir,
            video_info,
            analysis_params,
            mock_deps["progress_queue"],
            mock_deps["cancel_event"],
            mock_deps["logger"],
            mock_deps["config"],
        )

    assert mock_popen_cls.called
    cmd = mock_popen_cls.call_args[0][0]
    assert "ffmpeg" in cmd
    assert "-i" in cmd
    assert str(output_dir / "thumbs") in cmd[-1]


@patch("core.managers.extraction.is_image_folder", return_value=True)
@patch("core.managers.extraction.ingest_folder")
def test_extraction_pipeline_image_folder(mock_ingest, mock_is_img, mock_deps, analysis_params, tmp_path):
    analysis_params.source_path = str(tmp_path / "images")
    analysis_params.output_folder = str(tmp_path / "output")

    mock_ingest.return_value = [{"source": "img1.jpg", "preview": "preview1.jpg"}]

    pipeline = ExtractionPipeline(
        mock_deps["config"],
        mock_deps["logger"],
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
    )

    with (
        patch("core.managers.extraction.shutil.copy2"),
        patch("core.managers.extraction.Path.exists", return_value=True),
    ):
        result = pipeline.run()

    assert result["done"] is True
    assert (tmp_path / "output" / "scenes.json").exists()
    assert (tmp_path / "output" / "frame_map.json").exists()


@patch("core.managers.extraction.VideoManager")
@patch("core.managers.extraction.run_ffmpeg_extraction")
def test_extraction_pipeline_video(mock_run_ffmpeg, mock_vid_manager_cls, mock_deps, analysis_params, tmp_path):
    analysis_params.source_path = "video.mp4"
    analysis_params.output_folder = str(tmp_path / "output")

    mock_vid_manager = mock_vid_manager_cls.return_value
    mock_vid_manager.prepare_video.return_value = "video.mp4"
    mock_vid_manager_cls.get_video_info.return_value = {"fps": 30, "frame_count": 300}

    pipeline = ExtractionPipeline(
        mock_deps["config"],
        mock_deps["logger"],
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
    )

    result = pipeline.run()

    assert result["done"] is True
    assert mock_run_ffmpeg.called


@patch("core.managers.extraction.subprocess.Popen")
@patch("core.managers.extraction.detect_hwaccel", create=True)
def test_run_ffmpeg_extraction_resume(mock_detect, mock_popen_cls, mock_deps, analysis_params, tmp_path):
    mock_detect.return_value = (None, None)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "thumbs").mkdir()

    # Mock existing frame map
    with open(output_dir / "frame_map.json", "w") as f:
        json.dump([1, 11], f)

    # Create a dummy thumb file to trigger resume
    (output_dir / "thumbs" / "frame_000001.webp").touch()

    analysis_params.resume = True
    video_info = {"fps": 30, "frame_count": 300}

    mock_process = mock_popen_cls.return_value
    mock_process.stdout = MagicMock()
    mock_process.stdout.readline.return_value = ""
    mock_process.stderr = MagicMock()
    mock_process.stderr.readline.return_value = ""
    mock_process.poll.return_value = 0

    with patch("core.managers.extraction.subprocess.run"):
        run_ffmpeg_extraction(
            "video.mp4",
            output_dir,
            video_info,
            analysis_params,
            mock_deps["progress_queue"],
            mock_deps["cancel_event"],
            mock_deps["logger"],
            mock_deps["config"],
        )

    assert "-start_number" in mock_popen_cls.call_args[0][0]
    # start_frame_idx should be 1, so start_number should be 2
    assert "2" in mock_popen_cls.call_args[0][0]


@patch("core.managers.extraction.subprocess.Popen")
def test_run_ffmpeg_extraction_cancellation(mock_popen_cls, mock_deps, analysis_params, tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    video_info = {"fps": 30, "frame_count": 300}

    mock_process = mock_popen_cls.return_value
    mock_process.stdout = MagicMock()
    mock_process.stdout.readline.return_value = ""
    mock_process.stderr = MagicMock()
    mock_process.stderr.readline.return_value = ""

    # poll() returns None twice, then we set cancel_event, then it returns 0 after terminate
    mock_process.poll.side_effect = [None, None, 0]

    # We need to simulate the wait timing out to check cancel_event
    mock_process.wait.side_effect = subprocess.TimeoutExpired(["cmd"], 0.1)

    def set_cancel(*args, **kwargs):
        mock_deps["cancel_event"].set()
        # After one timeout, we allow poll to return 0 to exit loop
        mock_process.poll.side_effect = [None, 0]

    mock_process.wait.side_effect = [subprocess.TimeoutExpired(["cmd"], 0.1), 0]

    # Simpler: just set it before and ensure wait is called
    mock_deps["cancel_event"].set()

    run_ffmpeg_extraction(
        "video.mp4",
        output_dir,
        video_info,
        analysis_params,
        mock_deps["progress_queue"],
        mock_deps["cancel_event"],
        mock_deps["logger"],
        mock_deps["config"],
    )

    assert mock_process.terminate.called
