import threading
from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from core.events import ExtractionEvent
from core.models import AnalysisParameters, Scene
from core.pipelines import (
    AnalysisPipeline,
    ExtractionPipeline,
    _process_ffmpeg_showinfo,
    _process_ffmpeg_stream,
    execute_extraction,
    execute_session_load,
    run_ffmpeg_extraction,
)


class TestPipelines:
    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    @pytest.fixture
    def mock_config(self, tmp_path):
        config = MagicMock()
        config.ffmpeg_thumbnail_quality = 80
        config.downloads_dir = str(tmp_path / "downloads")
        config.models_dir = str(tmp_path / "models")
        config.analysis_default_workers = 1
        config.analysis_default_batch_size = 1
        config.retry_max_attempts = 1
        config.retry_backoff_seconds = (0.1,)
        config.sharpness_base_scale = 1.0
        config.edge_strength_base_scale = 1.0
        config.monitoring_memory_warning_threshold_mb = 1000
        config.validation_min_duration_secs = 0.1
        config.validation_min_frame_count = 1
        return config

    @pytest.fixture
    def mock_params(self, tmp_path):
        return AnalysisParameters(
            source_path="test.mp4",
            output_folder=str(tmp_path / "output"),
            method="interval",
            interval=1.0,
            thumbnails_only=True,
            scene_detect=False,
            video_path="test.mp4",
            max_resolution="1080",
            thumb_megapixels=1.0,
            nth_frame=1,
        )

    @pytest.fixture
    def mock_queue(self):
        return Queue()

    @pytest.fixture
    def mock_cancel_event(self):
        return threading.Event()

    # --- FFmpeg Helper Tests ---

    def test_process_ffmpeg_stream(self):
        stream = MagicMock()
        stream.readline.side_effect = ["frame=100", "out_time_us=1000000", "progress=continue", "progress=end", ""]
        tracker = MagicMock()
        tracker.total = 100

        _process_ffmpeg_stream(stream, tracker, "Desc", 10.0)

        tracker.set.assert_called()
        stream.close.assert_called_once()

    def test_process_ffmpeg_showinfo(self):
        stream = MagicMock()
        stream.readline.side_effect = [
            "[Parsed_showinfo_2 @ 0x...] n:   0 pts:      0 pts_time:0       pos:      0 fmt:rgb24 sar:1/1 s:100x100 i:P iskey:1 type:I checksum:...",
            "[Parsed_showinfo_2 @ 0x...] n:   1 pts:      1 ...",
            "",
        ]

        frames, stderr = _process_ffmpeg_showinfo(stream)

        assert frames == [0, 1]
        assert "n:   0" in stderr
        stream.close.assert_called_once()

    # --- ExtractionPipeline Tests ---

    @patch("subprocess.Popen")
    def test_run_ffmpeg_extraction(
        self, mock_popen, mock_params, mock_queue, mock_cancel_event, mock_logger, mock_config, tmp_path
    ):
        video_info = {"width": 100, "height": 100, "fps": 30, "frame_count": 100}

        process_mock = mock_popen.return_value
        process_mock.poll.side_effect = [None, 0]  # Run once then finish
        process_mock.returncode = 0
        process_mock.stdout.readline.return_value = ""  # EOF immediately
        process_mock.stderr.readline.return_value = ""

        run_ffmpeg_extraction(
            "test.mp4", tmp_path, video_info, mock_params, mock_queue, mock_cancel_event, mock_logger, mock_config
        )

        assert mock_popen.called
        # Check that subprocess.Popen was called at least once
        # Note: run_ffmpeg_extraction calls Popen for extraction, AND subprocess.run for downscaling.
        # But we mocked subprocess.Popen.

        # It seems the test is failing because the cmd being asserted is the one for downscaling (subprocess.run might be calling Popen internally or mocked too? No, patch('subprocess.Popen') only patches Popen).
        # Wait, the failure message shows:
        # ['ffmpeg', ..., '/video_lowres.mp4']
        # This is the downscaling command.
        # It means `run_ffmpeg_extraction` calls Popen twice (or Popen once and run once, and run uses Popen).
        # Since we patched Popen, both calls are intercepted if `subprocess.run` uses `subprocess.Popen` internally (which it does).

        # So we should check if ANY of the calls to Popen contained the thumbnail path.

        found_thumbs = False
        for call_args in mock_popen.call_args_list:
            cmd = call_args[0][0]
            for arg in cmd:
                if "frame_%06d.webp" in str(arg):
                    found_thumbs = True
                    break
            if found_thumbs:
                break

        assert found_thumbs, f"Thumbnail output path not found in any Popen call. Calls: {mock_popen.call_args_list}"

    @patch("core.pipelines.run_ffmpeg_extraction")
    @patch("core.managers.VideoManager")
    def test_extraction_pipeline_run_video(
        self, mock_vm_cls, mock_ffmpeg, mock_params, mock_queue, mock_cancel_event, mock_logger, mock_config
    ):
        pipeline = ExtractionPipeline(mock_config, mock_logger, mock_params, mock_queue, mock_cancel_event)

        # We need to ensure that when _run_impl calls `from core.utils import is_image_folder`, it gets our mock.
        # But `is_image_folder` is imported INSIDE `_run_impl`.
        # Patching `core.utils.is_image_folder` GLOBALLY should work because it's imported at runtime.

        with patch("core.pipelines.is_image_folder", create=True):
            # Wait, `is_image_folder` is imported from `core.utils` inside the function.
            # So `core.pipelines.is_image_folder` does NOT exist at module level.
            # We must patch `core.utils.is_image_folder`.
            pass

        with patch("core.utils.is_image_folder", return_value=False):
            # We also need to patch VideoManager in core.pipelines because it is imported at module level.
            with patch("core.pipelines.VideoManager") as mock_vm_cls_pipeline:
                mock_vm = mock_vm_cls_pipeline.return_value
                mock_vm.prepare_video.return_value = "prepared.mp4"
                mock_vm_cls_pipeline.get_video_info.return_value = {"fps": 30, "frame_count": 100}

                res = pipeline.run()

        assert res["done"] is True
        mock_vm.prepare_video.assert_called()
        mock_ffmpeg.assert_called()

    @patch("core.pipelines.make_photo_thumbs")
    def test_extraction_pipeline_run_folder(
        self, mock_make_thumbs, mock_params, mock_queue, mock_cancel_event, mock_logger, mock_config
    ):
        pipeline = ExtractionPipeline(mock_config, mock_logger, mock_params, mock_queue, mock_cancel_event)

        # Patching `core.utils.list_images` and `core.utils.is_image_folder`
        with (
            patch("core.utils.list_images", return_value=[Path("img1.jpg")]),
            patch("core.utils.is_image_folder", return_value=True),
        ):
            res = pipeline.run()

        assert res["done"] is True
        mock_make_thumbs.assert_called()
        assert (Path(mock_params.output_folder) / "scenes.json").exists()

    # --- AnalysisPipeline Tests ---

    @patch("core.pipelines.SubjectMasker")
    @patch("core.pipelines.initialize_analysis_models")
    @patch("core.pipelines.create_frame_map")
    def test_run_full_analysis(
        self,
        mock_frame_map,
        mock_init_models,
        mock_masker_cls,
        mock_params,
        mock_queue,
        mock_cancel_event,
        mock_logger,
        mock_config,
        tmp_path,
    ):
        # Setup
        thumbnail_manager = MagicMock()
        model_registry = MagicMock()

        # Ensure output folder exists for DB
        output_folder = tmp_path / "output"
        output_folder.mkdir()
        mock_params.output_folder = str(output_folder)

        pipeline = AnalysisPipeline(
            mock_config, mock_logger, mock_params, mock_queue, mock_cancel_event, thumbnail_manager, model_registry
        )

        scenes = [Scene(shot_id=1, start_frame=0, end_frame=10)]

        # Mocks
        mock_init_models.return_value = {
            "face_analyzer": None,
            "ref_emb": None,
            "face_landmarker": None,
            "device": "cpu",
        }
        mock_masker = mock_masker_cls.return_value
        mock_masker.run_propagation.return_value = {}

        # We need a real DB or mocked DB.
        # "unable to open database file" usually happens when directory doesn't exist.

        res = pipeline.run_full_analysis(scenes)

        # Check for exceptions logged
        if not res["done"]:
            print(f"Failed with log: {res.get('log')} or error: {res.get('error')}")

        assert res["done"] is True
        mock_masker.run_propagation.assert_called()

    @patch("core.pipelines.create_frame_map")
    @patch("core.pipelines.initialize_analysis_models")
    def test_run_analysis_only(
        self, mock_init, mock_frame_map, mock_params, mock_queue, mock_cancel_event, mock_logger, mock_config, tmp_path
    ):
        # Setup
        thumbnail_manager = MagicMock()
        model_registry = MagicMock()

        # Ensure output folder exists for DB
        output_folder = tmp_path / "output_ana"
        output_folder.mkdir()
        mock_params.output_folder = str(output_folder)

        pipeline = AnalysisPipeline(
            mock_config, mock_logger, mock_params, mock_queue, mock_cancel_event, thumbnail_manager, model_registry
        )

        scenes = [Scene(shot_id=1, start_frame=0, end_frame=1)]

        mock_init.return_value = {"face_analyzer": None, "ref_emb": None, "face_landmarker": None, "device": "cpu"}
        mock_frame_map.return_value = {0: "frame_0.webp"}
        thumbnail_manager.get.return_value = np.zeros((10, 10, 3), dtype=np.uint8)

        res = pipeline.run_analysis_only(scenes)

        if not res["done"]:
            print(f"Failed with log: {res.get('log')} or error: {res.get('error')}")

        assert res["done"] is True
        # Verify metadata was inserted
        # Since we use real DB here (sqlite), we can check file
        assert (output_folder / "metadata.db").exists()

    # --- Execute Helpers Tests ---

    @patch("core.pipelines.ExtractionPipeline")
    @patch("core.pipelines.shutil.copy2")
    def test_execute_extraction(self, mock_copy, mock_pipeline_cls, mock_logger, mock_config):
        event = ExtractionEvent(
            source_path="src.mp4",
            upload_video="upload.mp4",
            method="interval",
            interval=1.0,
            thumbnails_only=True,
            max_resolution="1080",
            thumb_megapixels=1.0,
            nth_frame=1,
            scene_detect=False,
        )

        # Mock pipeline run
        mock_inst = mock_pipeline_cls.return_value
        mock_inst.run.return_value = {"done": True, "output_dir": "/tmp/out", "video_path": "/tmp/vid.mp4"}

        gen = execute_extraction(event, Queue(), threading.Event(), mock_logger, mock_config)

        res = next(gen)

        assert res["done"] is True
        mock_copy.assert_called()

    def test_validate_session_dir(self, tmp_path):
        from core.pipelines import validate_session_dir

        path, err = validate_session_dir(str(tmp_path))
        assert path == tmp_path
        assert err is None

        path, err = validate_session_dir(str(tmp_path / "nonexistent"))
        assert path is None
        assert err is not None

    def test_execute_session_load_invalid(self, mock_logger):
        res = execute_session_load(MagicMock(session_path=""), mock_logger)
        assert "error" in res

    def test_execute_session_load_valid(self, mock_logger, tmp_path):
        session_path = tmp_path
        (session_path / "run_config.json").write_text("{}")

        event = MagicMock(session_path=str(session_path))

        res = execute_session_load(event, mock_logger)

        assert res["success"] is True
        assert res["metadata_exists"] is False
