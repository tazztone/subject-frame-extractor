import textwrap
import threading
from io import StringIO
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
def mock_logger():
    return MagicMock()


@pytest.fixture
def mock_config(tmp_path):
    config = MagicMock()
    config.logs_dir = str(tmp_path / "logs")
    config.downloads_dir = str(tmp_path / "downloads")
    config.ffmpeg_hwaccel = "off"
    config.ffmpeg_thumbnail_quality = 80
    config.retry_max_attempts = 1
    config.retry_backoff_seconds = 0
    return config


def test_process_ffmpeg_stream(mock_logger, mock_config):
    mock_stream = StringIO(
        textwrap.dedent("""\
        frame=10
        out_time_us=333333
        progress=continue
        frame=20
        out_time_us=666666
        progress=end
    """)
    )
    tracker = MagicMock()
    tracker.total = 100
    _process_ffmpeg_stream(mock_stream, tracker, "Extracting", 1.0)
    assert tracker.set.call_count == 3
    assert tracker.set.call_args_list[0][0][0] == 33
    assert tracker.set.call_args_list[1][0][0] == 66
    assert tracker.set.call_args_list[2][0][0] == 100


def test_process_ffmpeg_showinfo(mock_logger):
    mock_stream = StringIO(
        textwrap.dedent("""\
        [Parsed_showinfo_0] n: 0 pts_time:0
        [Parsed_showinfo_0] n: 1 pts_time:0.033366
        [Parsed_showinfo_0] n: 2 pts_time:0.066733
    """)
    )
    frame_numbers, stderr = _process_ffmpeg_showinfo(mock_stream, 30.0)
    assert frame_numbers == [0, 1, 2]


@patch("core.managers.extraction.subprocess.Popen")
@patch("core.utils.detect_hwaccel", return_value=(None, None))
def test_run_ffmpeg_extraction(mock_detect, mock_popen, mock_logger, mock_config, tmp_path):
    mock_process = MagicMock()
    mock_process.stdout.readline.side_effect = ["progress=end", ""]
    mock_process.stderr.readline.side_effect = [""]
    mock_process.poll.return_value = 0
    mock_popen.return_value = mock_process
    with patch("core.managers.extraction.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        params = AnalysisParameters(source_path="video.mp4", method="all", thumb_megapixels=0.5, thumbnails_only=True)
        run_ffmpeg_extraction(
            "video.mp4",
            tmp_path,
            {"fps": 30, "frame_count": 300},
            params,
            Queue(),
            threading.Event(),
            mock_logger,
            mock_config,
        )
    assert mock_popen.called


@patch("core.managers.extraction.run_ffmpeg_extraction")
@patch("core.managers.extraction.VideoManager")
def test_extraction_pipeline_run_video(mock_vm_cls, mock_run_ffmpeg, mock_logger, mock_config, tmp_path):
    mock_vm = mock_vm_cls.return_value
    mock_vm.prepare_video.return_value = "video.mp4"
    mock_vm_cls.get_video_info.return_value = {"fps": 30, "frame_count": 300}
    params = AnalysisParameters(source_path="video.mp4", output_folder=str(tmp_path))
    pipeline = ExtractionPipeline(mock_config, mock_logger, params, Queue(), threading.Event())
    res = pipeline._run_impl()
    assert res["done"] is True
    assert res["video_path"] == "video.mp4"


@patch("core.managers.extraction.ingest_folder")
@patch("core.managers.extraction.is_image_folder", return_value=True)
def test_extraction_pipeline_run_folder(mock_is_img, mock_ingest, mock_logger, mock_config, tmp_path):
    mock_ingest.return_value = [{"id": "1", "source": "1.jpg", "preview": "1.jpg"}]
    params = AnalysisParameters(source_path="photos", output_folder=str(tmp_path))
    pipeline = ExtractionPipeline(mock_config, mock_logger, params, Queue(), threading.Event())
    res = pipeline._run_impl()
    assert res["done"] is True
    assert (tmp_path / "scenes.json").exists()
