import threading
from queue import Queue
from unittest.mock import MagicMock, patch

import pytest

from core.events import ExtractionEvent
from core.models import AnalysisParameters
from core.pipelines import AnalysisPipeline, ExtractionPipeline


class TestPipelines:
    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    @pytest.fixture
    def mock_config(self, tmp_path):
        config = MagicMock()
        config.logs_dir = str(tmp_path / "logs")
        config.downloads_dir = str(tmp_path / "downloads")
        config.validation_min_duration_secs = 0.1
        config.validation_min_frame_count = 1
        config.analysis_default_workers = 1
        config.analysis_default_batch_size = 1
        return config

    def test_process_ffmpeg_stream(self, mock_logger, mock_config):
        import textwrap
        from io import StringIO

        from core.pipelines import _process_ffmpeg_stream

        mock_stream = StringIO(textwrap.dedent("""\
            frame=10
            fps=30.00
            stream_0_0_q=28.0
            bitrate=N/A
            total_size=N/A
            out_time_us=333333
            out_time_ms=333333
            out_time=00:00:00.333333
            dup_frames=0
            drop_frames=0
            speed=N/A
            progress=continue
            frame=20
            fps=30.00
            stream_0_0_q=28.0
            bitrate=N/A
            total_size=N/A
            out_time_us=666666
            out_time_ms=666666
            out_time=00:00:00.666666
            dup_frames=0
            drop_frames=0
            speed=N/A
            progress=end
        """))

        tracker = MagicMock()
        tracker.total = 100

        _process_ffmpeg_stream(mock_stream, tracker, "Extracting", 1.0)

        # Test out_time_us parsing
        assert tracker.set.call_count == 3
        # First call: out_time_us=333333 -> 0.333333s / 1.0s = 0.33 -> 33
        assert tracker.set.call_args_list[0][0][0] == 33
        # Second call: out_time_us=666666 -> 0.666666s / 1.0s = 0.66 -> 66
        assert tracker.set.call_args_list[1][0][0] == 66
        # Third call: progress=end
        assert tracker.set.call_args_list[2][0][0] == 100

    def test_process_ffmpeg_stream_no_duration(self, mock_logger, mock_config):
        import textwrap
        from io import StringIO

        from core.pipelines import _process_ffmpeg_stream

        mock_stream = StringIO(textwrap.dedent("""\
            frame=10
            fps=30.00
            progress=continue
            frame=20
            fps=30.00
            progress=end
        """))

        tracker = MagicMock()
        tracker.total = 100

        _process_ffmpeg_stream(mock_stream, tracker, "Extracting", 0.0)

        # Test frame parsing
        assert tracker.set.call_count == 3
        # First call: frame=10
        assert tracker.set.call_args_list[0][0][0] == 10
        # Second call: frame=20
        assert tracker.set.call_args_list[1][0][0] == 20
        # Third call: progress=end
        assert tracker.set.call_args_list[2][0][0] == 100

    def test_process_ffmpeg_showinfo(self, mock_logger):
        import textwrap
        from io import StringIO

        from core.pipelines import _process_ffmpeg_showinfo

        mock_stream = StringIO(textwrap.dedent("""\
            [Parsed_showinfo_0 @ 0x7f8b9c004400] n:   0 pts:      0 pts_time:0       pos:      123 fmt:yuv420p sar:1/1 s:1920x1080 i:P iskey:1 type:I checksum:1A2B3C4D plane_checksum:[1A2B3C4D] mean:[128 128 128] stdev:[50 50 50]
            [Parsed_showinfo_0 @ 0x7f8b9c004400] n:   1 pts:   1001 pts_time:0.033366 pos:     4567 fmt:yuv420p sar:1/1 s:1920x1080 i:P iskey:0 type:P checksum:5E6F7G8H plane_checksum:[5E6F7G8H] mean:[128 128 128] stdev:[50 50 50]
            [Parsed_showinfo_0 @ 0x7f8b9c004400] n:   2 pts:   2002 pts_time:0.066733 pos:     8901 fmt:yuv420p sar:1/1 s:1920x1080 i:P iskey:0 type:P checksum:9I0J1K2L plane_checksum:[9I0J1K2L] mean:[128 128 128] stdev:[50 50 50]
            Some random error line
        """))

        frame_numbers, stderr_output = _process_ffmpeg_showinfo(mock_stream, 30.0)

        # 0 * 30 = 0
        # 0.033366 * 30 = 1.00098 -> 1
        # 0.066733 * 30 = 2.00199 -> 2
        assert frame_numbers == [0, 1, 2]
        assert "Some random error line" in stderr_output

    @patch("core.pipelines.subprocess.Popen")
    @patch("core.utils.detect_hwaccel", return_value=(None, None))
    def test_run_ffmpeg_extraction(self, mock_detect, mock_popen, mock_logger, mock_config, tmp_path):
        import threading

        from core.pipelines import run_ffmpeg_extraction

        mock_config.ffmpeg_hwaccel = "auto"
        mock_config.ffmpeg_thumbnail_quality = 80

        mock_process = MagicMock()
        mock_process.stdout = MagicMock()
        mock_process.stderr = MagicMock()
        mock_process.stdout.__enter__.return_value = mock_process.stdout
        mock_process.stderr.__enter__.return_value = mock_process.stderr
        mock_process.stdout.readline.side_effect = ["progress=end", ""]
        mock_process.stderr.readline.side_effect = [""]
        mock_process.poll.return_value = 0
        mock_process.returncode = 0

        # Mock downscale video run
        with patch("core.pipelines.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            mock_popen.return_value = mock_process

            params = AnalysisParameters(
                source_path="video.mp4",
                method="every_nth_frame",
                nth_frame=5,
                thumb_megapixels=0.5,
                thumbnails_only=True
            )

            run_ffmpeg_extraction(
                "video.mp4",
                tmp_path,
                {"width": 1920, "height": 1080, "fps": 30, "frame_count": 300},
                params,
                Queue(),
                threading.Event(),
                mock_logger,
                mock_config
            )

        assert mock_popen.called
        assert mock_run.called

    @patch("core.pipelines.run_ffmpeg_extraction")
    @patch("core.pipelines.VideoManager.get_video_info")
    @patch("core.managers.video.validate_video_file", return_value="video.mp4")
    def test_extraction_pipeline_run_video(
        self, mock_val, mock_info, mock_run_ffmpeg, mock_logger, mock_config, tmp_path
    ):
        mock_info.return_value = {"fps": 30, "frame_count": 300}

        event = ExtractionEvent(
            source_path="video.mp4",
            method="all",
            interval=1.0,
            nth_frame=1,
            max_resolution="720",
            thumbnails_only=True,
            thumb_megapixels=0.5,
            scene_detect=False,
            output_folder=str(tmp_path),
        )

        pipeline = ExtractionPipeline(mock_config, mock_logger, event, Queue(), threading.Event())
        res = pipeline.run()

        assert res["done"] is True
        assert res["video_path"] == "video.mp4"

    @patch("core.pipelines.ingest_folder")
    def test_extraction_pipeline_run_folder(self, mock_ingest, mock_logger, mock_config, tmp_path):
        # Create a real dummy folder
        photos_dir = tmp_path / "photos"
        photos_dir.mkdir()

        mock_ingest.return_value = [{"id": "1", "source": "1.jpg", "preview": "1.jpg"}]

        event = ExtractionEvent(
            source_path=str(photos_dir),  # MUST be an existing dir
            method="all",
            interval=1.0,
            nth_frame=1,
            max_resolution="720",
            thumbnails_only=True,
            thumb_megapixels=0.5,
            scene_detect=False,
            output_folder=str(tmp_path),
        )

        pipeline = ExtractionPipeline(mock_config, mock_logger, event, Queue(), threading.Event())
        res = pipeline.run()

        assert res["done"] is True
        assert res["output_dir"] == str(tmp_path)

    @patch("core.pipelines.initialize_analysis_models")
    @patch("core.pipelines.SubjectMasker")
    @patch("core.pipelines.create_frame_map")
    @patch("core.pipelines.Database")
    def test_run_full_analysis(self, mock_db, mock_cfm, mock_masker, mock_init, mock_logger, mock_config, tmp_path):
        import json

        from core.models import Scene

        # Setup mocks
        mock_init.return_value = {"face_analyzer": None, "ref_emb": None, "face_landmarker": None, "device": "cpu"}
        mock_db_instance = MagicMock()
        mock_db.return_value = mock_db_instance

        # Setup params and scene
        params = AnalysisParameters(
            source_path="video.mp4",
            video_path="video.mp4",  # Add video_path to skip folder logic
            output_folder=str(tmp_path),
            resume=False
        )
        scene = Scene(shot_id=1, start_frame=0, end_frame=10)

        # Setup pipeline
        tm = MagicMock()
        registry = MagicMock()
        pipeline = AnalysisPipeline(mock_config, mock_logger, params, Queue(), threading.Event(), tm, registry)

        # Mock propagation result
        mock_masker_instance = mock_masker.return_value
        mock_masker_instance.run_propagation.return_value = {"frame_000001.webp": {"mask_path": "mask.png"}}

        # Run
        res = pipeline.run_full_analysis([scene])

        # Verify
        assert res["done"] is True
        assert res["output_dir"] == str(tmp_path)
        mock_init.assert_called_once()
        mock_masker_instance.run_propagation.assert_called_once()

        # Verify progress saving
        progress_file = tmp_path / "progress.json"
        assert progress_file.exists()
        with open(progress_file, "r") as f:
            data = json.load(f)
            assert 1 in data["completed_scenes"]

    @patch("core.pipelines.initialize_analysis_models")
    def test_run_analysis_only(self, mock_init, mock_logger, mock_config, tmp_path):
        # AnalysisPipeline needs frame_map.json
        (tmp_path / "frame_map.json").write_text("{}")

        params = AnalysisParameters(output_folder=str(tmp_path))
        tm = MagicMock()
        registry = MagicMock()

        pipeline = AnalysisPipeline(mock_config, mock_logger, params, Queue(), threading.Event(), tm, registry)

        with patch.object(pipeline, "_run_analysis_loop"):
            res = pipeline.run_analysis_only([])
            assert res["done"] is True

    def test_execute_extraction(self, mock_config, mock_logger, tmp_path):
        pass

    def test_validate_session_dir(self, tmp_path):
        pass

    @patch("core.pipelines.shutil.copy2")
    def test_handle_extraction_uploads(self, mock_copy, mock_config):
        from core.pipelines import _handle_extraction_uploads

        event_dict = {"upload_video": "/tmp/upload.mp4"}
        mock_config.downloads_dir = "/tmp/downloads"

        res = _handle_extraction_uploads(event_dict, mock_config)

        mock_copy.assert_called_once_with("/tmp/upload.mp4", "/tmp/downloads/upload.mp4")
        assert res["source_path"] == "/tmp/downloads/upload.mp4"
        assert "upload_video" not in res

    def test_initialize_extraction_params(self, mock_config, mock_logger):
        from core.pipelines import _initialize_extraction_params

        event_dict = {"source_path": "/tmp/test.mp4", "method": "keyframes"}

        with patch("core.models.AnalysisParameters.from_ui") as mock_from_ui:
            mock_from_ui.return_value = "mock_params"
            res = _initialize_extraction_params(event_dict, mock_config, mock_logger)

            mock_from_ui.assert_called_once_with(mock_logger, mock_config, **event_dict)
            assert res == "mock_params"
