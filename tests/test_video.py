import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import sys

# Mock heavy dependencies
sys.modules['cv2'] = MagicMock()
sys.modules['yt_dlp'] = MagicMock()
sys.modules['scenedetect'] = MagicMock()

from app.video import VideoManager, run_scene_detection, run_ffmpeg_extraction

class TestVideoManager(unittest.TestCase):

    @patch('app.video.Config')
    def test_prepare_video_local_file(self, mock_config):
        mock_config.return_value.DIRS = {'downloads': Path('/tmp/downloads')}
        # Test with a local file that exists
        with patch('pathlib.Path.is_file', return_value=True):
            manager = VideoManager('/fake/video.mp4')
            result = manager.prepare_video()
            self.assertEqual(result, '/fake/video.mp4')

        # Test with a local file that does not exist
        with patch('pathlib.Path.is_file', return_value=False):
            manager = VideoManager('/fake/nonexistent.mp4')
            with self.assertRaises(FileNotFoundError):
                manager.prepare_video()

    @patch('app.video.ytdlp.YoutubeDL')
    @patch('app.video.Config')
    def test_prepare_video_youtube(self, mock_config, mock_ytdl):
        mock_config.return_value.DIRS = {'downloads': Path('/tmp/downloads')}
        mock_instance = mock_ytdl.return_value.__enter__.return_value
        mock_instance.extract_info.return_value = {'id': 'test_id', 'title': 'test_title', 'height': 1080, 'ext': 'mp4'}
        mock_instance.prepare_filename.return_value = '/tmp/downloads/test_id_test_title_1080p.mp4'

        manager = VideoManager('https://www.youtube.com/watch?v=test_id')
        result = manager.prepare_video()
        self.assertEqual(result, '/tmp/downloads/test_id_test_title_1080p.mp4')
        mock_ytdl.assert_called_once()
        mock_instance.extract_info.assert_called_once_with('https://www.youtube.com/watch?v=test_id', download=True)

    @patch('app.video.cv2.VideoCapture')
    def test_get_video_info(self, mock_videocapture):
        mock_cap_instance = mock_videocapture.return_value
        mock_cap_instance.isOpened.return_value = True
        mock_cap_instance.get.side_effect = [1920, 1080, 30, 1000]

        info = VideoManager.get_video_info('/fake/video.mp4')

        self.assertEqual(info['width'], 1920)
        self.assertEqual(info['height'], 1080)
        self.assertEqual(info['fps'], 30)
        self.assertEqual(info['frame_count'], 1000)
        mock_cap_instance.release.assert_called_once()

class TestSceneDetection(unittest.TestCase):

    @patch('app.video.detect')
    @patch('pathlib.Path.open', new_callable=mock_open)
    @patch('json.dump')
    def test_run_scene_detection_success(self, mock_json_dump, mock_path_open, mock_detect):
        # Mock the scene detection to return a list of scenes
        mock_scene = (MagicMock(), MagicMock())
        mock_scene[0].frame_num = 0
        mock_scene[1].frame_num = 100
        mock_detect.return_value = [mock_scene]

        # Pass a mock logger to prevent psutil issues
        logger_mock = MagicMock()
        shots = run_scene_detection('/fake/video.mp4', Path('/tmp/output'), logger=logger_mock)

        # Assertions
        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0], (0, 100))

        # Check that the function behaved as expected
        mock_detect.assert_called_once()
        # Path.open is called on the instance, so the mock doesn't see the path itself
        mock_path_open.assert_called_once_with('w', encoding='utf-8')
        mock_json_dump.assert_called_once_with([(0, 100)], mock_path_open())

        # Check logging
        logger_mock.info.assert_called_with("Detecting scenes...", component="video")
        logger_mock.success.assert_called_with("Found 1 scenes.", component="video")
        logger_mock.error.assert_not_called()

    @patch('app.video.detect', side_effect=Exception("Detection failed"))
    def test_run_scene_detection_failure(self, mock_detect):
        logger_mock = MagicMock()
        shots = run_scene_detection('/fake/video.mp4', Path('/tmp/output'), logger=logger_mock)
        self.assertEqual(shots, [])
        logger_mock.error.assert_called_once_with("Scene detection failed.", component="video", exc_info=True)


if __name__ == '__main__':
    unittest.main()