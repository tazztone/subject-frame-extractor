import unittest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
from core.photo_utils import extract_preview

class TestPhase2Logic(unittest.TestCase):
    @patch('core.photo_utils.shutil.which')
    @patch('core.photo_utils.subprocess.run')
    @patch('core.photo_utils.Image.open')
    def test_extract_preview_priority_thumbnails_only(self, mock_image_open, mock_run, mock_which):
        mock_which.return_value = '/usr/bin/exiftool'
        
        # Mock subprocess.run: 
        # PreviewImage returns too small data (e.g. 1KB)
        # ThumbnailImage returns enough data (e.g. 30KB)
        def side_effect(cmd, **kwargs):
            if "-PreviewImage" in cmd:
                return MagicMock(returncode=0, stdout=b"x" * 1024) # 1KB
            if "-ThumbnailImage" in cmd:
                return MagicMock(returncode=0, stdout=b"x" * 30 * 1024) # 30KB
            return MagicMock(returncode=1, stdout=b"")
        
        mock_run.side_effect = side_effect
        
        # Mock PIL Image
        mock_img = MagicMock()
        mock_img.size = (800, 600)
        mock_image_open.return_value.__enter__.return_value = mock_img
        
        with patch('core.photo_utils.open', mock_open()) as mocked_file:
            with patch('core.photo_utils.Path.exists', side_effect=[True, False]):
                res = extract_preview(Path("test.ARW"), Path("out"), thumbnails_only=True)
                
                self.assertIsNotNone(res)
                # Check that PreviewImage was the first tag tried
                first_call_args = mock_run.call_args_list[0][0][0]
                self.assertIn("-PreviewImage", first_call_args)
                
                # Check that it eventually took ThumbnailImage because PreviewImage was too small
                second_call_args = mock_run.call_args_list[1][0][0]
                self.assertIn("-ThumbnailImage", second_call_args)

    @patch('core.photo_utils.shutil.which')
    @patch('core.photo_utils.subprocess.run')
    @patch('core.photo_utils.Image.open')
    def test_extract_preview_resize(self, mock_image_open, mock_run, mock_which):
        mock_which.return_value = '/usr/bin/exiftool'
        # Return enough data to pass the 25KB check
        mock_run.return_value = MagicMock(returncode=0, stdout=b"x" * 30 * 1024)
        
        # Mock PIL Image that needs resizing
        mock_img = MagicMock()
        mock_img.size = (2000, 1500)
        mock_image_open.return_value.__enter__.return_value = mock_img
        
        with patch('core.photo_utils.open', mock_open()) as mocked_file:
            with patch('core.photo_utils.Path.exists', side_effect=[True, False]): 
                res = extract_preview(Path("test.ARW"), Path("out"), thumbnails_only=True)
                
                self.assertIsNotNone(res)
                # Check that thumbnail() was called
                mock_img.thumbnail.assert_called_once()
                # Check that save() was called with optimization
                mock_img.save.assert_called_once()

if __name__ == '__main__':
    unittest.main()