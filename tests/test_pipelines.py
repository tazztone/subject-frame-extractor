"""
Tests for pipeline functionality - extraction, analysis, and session loading.

Uses fixtures from conftest.py for mock setup.
"""
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from queue import Queue
import threading

from core.config import Config
from core.models import AnalysisParameters, Scene
from core.pipelines import ExtractionPipeline, AnalysisPipeline, run_ffmpeg_extraction


class TestExtractionPipeline:
    @patch('core.pipelines.run_ffmpeg_extraction')
    @patch('core.pipelines.VideoManager')
    def test_extraction_video_success(self, mock_vm_cls, mock_ffmpeg, mock_config_simple, mock_logger, mock_params, mock_progress_queue, mock_cancel_event):
        # Setup mocks
        mock_vm = mock_vm_cls.return_value
        mock_vm.prepare_video.return_value = Path("prepared_video.mp4")
        mock_vm_cls.get_video_info.return_value = {"duration": 10, "fps": 30}

        mock_ffmpeg.return_value = None

        pipeline = ExtractionPipeline(mock_config_simple, mock_logger, mock_params, mock_progress_queue, mock_cancel_event)

        # Run
        result = pipeline.run()

        # Assertions
        assert result['done'] is True
        mock_vm.prepare_video.assert_called_once()
        mock_ffmpeg.assert_called_once()

    @patch('core.pipelines.run_ffmpeg_extraction')
    @patch('core.pipelines.VideoManager')
    def test_extraction_video_cancel(self, mock_vm_cls, mock_ffmpeg, mock_config_simple, mock_logger, mock_params, mock_progress_queue, mock_cancel_event):
        mock_vm = mock_vm_cls.return_value
        mock_vm.prepare_video.return_value = Path("prepared_video.mp4")
        mock_vm_cls.get_video_info.return_value = {"duration": 10, "fps": 30}

        # Simulate cancel during ffmpeg
        def side_effect(*args, **kwargs):
            mock_cancel_event.set()
        mock_ffmpeg.side_effect = side_effect

        pipeline = ExtractionPipeline(mock_config_simple, mock_logger, mock_params, mock_progress_queue, mock_cancel_event)
        result = pipeline.run()

        assert result['done'] is False
        assert result['log'] == "Extraction cancelled"

    @patch('core.utils.is_image_folder', return_value=True)
    @patch('core.utils.list_images')
    @patch('core.pipelines.make_photo_thumbs')
    def test_extraction_folder(self, mock_thumbs, mock_list_imgs, mock_is_folder, mock_config_simple, mock_logger, mock_params, mock_progress_queue, mock_cancel_event):
        mock_list_imgs.return_value = [Path("img1.jpg"), Path("img2.jpg")]

        pipeline = ExtractionPipeline(mock_config_simple, mock_logger, mock_params, mock_progress_queue, mock_cancel_event)
        result = pipeline.run()

        assert result['done'] is True
        mock_thumbs.assert_called_once()
        # Check scenes.json
        output_dir = Path(result['output_dir'])
        assert (output_dir / "scenes.json").exists()

    @patch('subprocess.Popen')
    def test_run_ffmpeg_extraction(self, mock_popen, mock_config_simple, mock_logger, mock_params, mock_progress_queue, mock_cancel_event, tmp_path):
        # Setup mock process
        process = MagicMock()
        process.poll.side_effect = [None, 0]  # Run once then finish
        process.returncode = 0
        process.stdout.readline.return_value = ''
        process.stderr.readline.return_value = ''
        mock_popen.return_value = process

        video_info = {"width": 100, "height": 100, "fps": 30, "frame_count": 300}
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        run_ffmpeg_extraction("video.mp4", output_dir, video_info, mock_params, mock_progress_queue, mock_cancel_event, mock_logger, mock_config_simple)

        mock_popen.assert_called()
        args, _ = mock_popen.call_args
        assert "ffmpeg" in args[0]


class TestAnalysisPipeline:
    @patch('core.pipelines.SubjectMasker')
    @patch('core.pipelines.initialize_analysis_models')
    @patch('core.pipelines.Database')
    def test_run_full_analysis_success(self, mock_db_cls, mock_init_models, mock_masker_cls, mock_config_simple, mock_logger, mock_params, mock_progress_queue, mock_cancel_event, mock_thumbnail_manager, mock_model_registry):
        mock_db = mock_db_cls.return_value
        mock_init_models.return_value = {
            'face_analyzer': MagicMock(), 'ref_emb': MagicMock(), 'face_landmarker': MagicMock(), 'device': 'cpu'
        }
        mock_masker = mock_masker_cls.return_value
        mock_masker.run_propagation.return_value = {}

        pipeline = AnalysisPipeline(mock_config_simple, mock_logger, mock_params, mock_progress_queue, mock_cancel_event, mock_thumbnail_manager, mock_model_registry)

        scenes = [Scene(shot_id=1, start_frame=0, end_frame=10)]
        result = pipeline.run_full_analysis(scenes)

        assert result['done'] is True
        mock_masker.run_propagation.assert_called()

    @patch('core.pipelines.SubjectMasker')
    @patch('core.pipelines.initialize_analysis_models')
    def test_run_full_analysis_cancel(self, mock_init_models, mock_masker_cls, mock_config_simple, mock_logger, mock_params, mock_progress_queue, mock_cancel_event, mock_thumbnail_manager, mock_model_registry):
        mock_init_models.return_value = {
            'face_analyzer': MagicMock(), 'ref_emb': MagicMock(), 'face_landmarker': MagicMock(), 'device': 'cpu'
        }
        mock_cancel_event.set()  # Set cancel before running

        pipeline = AnalysisPipeline(mock_config_simple, mock_logger, mock_params, mock_progress_queue, mock_cancel_event, mock_thumbnail_manager, mock_model_registry)
        scenes = [Scene(shot_id=1, start_frame=0, end_frame=10)]

        result = pipeline.run_full_analysis(scenes)
        assert result['done'] is False
        assert "cancelled" in result.get('log', '').lower()

    @patch('core.pipelines.initialize_analysis_models')
    @patch('core.pipelines.Database')
    def test_run_analysis_only(self, mock_db_cls, mock_init_models, mock_config_simple, mock_logger, mock_params, mock_progress_queue, mock_cancel_event, mock_thumbnail_manager, mock_model_registry):
        mock_db = mock_db_cls.return_value
        mock_db.count_errors.return_value = 0
        mock_init_models.return_value = {
            'face_analyzer': MagicMock(), 'ref_emb': MagicMock(), 'face_landmarker': MagicMock(), 'device': 'cpu'
        }

        with patch('core.pipelines.create_frame_map', return_value={0: 'frame_000.webp', 1: 'frame_001.webp'}):
            with patch.object(AnalysisPipeline, '_process_batch', return_value=1) as mock_process_batch:
                pipeline = AnalysisPipeline(mock_config_simple, mock_logger, mock_params, mock_progress_queue, mock_cancel_event, mock_thumbnail_manager, mock_model_registry)
                pipeline.thumb_dir = Path("/tmp/out/thumbs")

                scenes = [Scene(shot_id=1, start_frame=0, end_frame=2)]
                result = pipeline.run_analysis_only(scenes)

                assert result['done'] is True
                mock_process_batch.assert_called()


class TestSessionLoad:
    @patch('core.pipelines.validate_session_dir')
    def test_execute_session_load_success(self, mock_validate, mock_logger, tmp_path):
        from core.pipelines import execute_session_load
        from core.events import SessionLoadEvent

        session_path = tmp_path / "session"
        session_path.mkdir()
        (session_path / "run_config.json").write_text('{"source_path": "foo"}', encoding='utf-8')
        (session_path / "scenes.json").write_text('[]', encoding='utf-8')

        mock_validate.return_value = (session_path, None)
        event = SessionLoadEvent(session_path=str(session_path))

        result = execute_session_load(event, mock_logger)

        assert result['success'] is True
        assert result['run_config']['source_path'] == "foo"
        assert result['metadata_exists'] is False

    @patch('core.pipelines.validate_session_dir')
    def test_execute_session_load_fail_validate(self, mock_validate, mock_logger):
        from core.pipelines import execute_session_load
        from core.events import SessionLoadEvent

        mock_validate.return_value = (None, "Invalid path")
        event = SessionLoadEvent(session_path="bad")

        result = execute_session_load(event, mock_logger)

        assert "error" in result
        assert result["error"] == "Invalid path"
