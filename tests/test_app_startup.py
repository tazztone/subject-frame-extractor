import unittest
from unittest.mock import MagicMock, patch
import os
import sys

# Mock torch and other heavy dependencies before they are imported by the app
sys.modules['torch'] = MagicMock()
sys.modules['torchvision'] = MagicMock()
sys.modules['torchvision.transforms'] = MagicMock()
sys.modules['torchvision.ops'] = MagicMock()
sys.modules['torchvision.transforms.functional'] = MagicMock()
sys.modules['gradio'] = MagicMock()
sys.modules['gradio.themes'] = MagicMock()
sys.modules['grounding_dino'] = MagicMock()
sys.modules['grounding_dino.util'] = MagicMock()
sys.modules['grounding_dino.util.inference'] = MagicMock()
sys.modules['grounding_dino.groundingdino'] = MagicMock()
sys.modules['grounding_dino.groundingdino.util'] = MagicMock()
sys.modules['grounding_dino.groundingdino.util.inference'] = MagicMock()
sys.modules['segment_anything'] = MagicMock()
sys.modules['ultralytics'] = MagicMock()
sys.modules['insightface'] = MagicMock()
sys.modules['insightface.app'] = MagicMock()
sys.modules['pyiqa'] = MagicMock()
sys.modules['imagehash'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['PIL.Image'] = MagicMock()
sys.modules['vot'] = MagicMock()
sys.modules['vot.region'] = MagicMock()
sys.modules['vot.region.raster'] = MagicMock()
sys.modules['vot.region.shapes'] = MagicMock()
sys.modules['sam2'] = MagicMock()
sys.modules['sam2.build_sam'] = MagicMock()
sys.modules['utils'] = MagicMock()
sys.modules['utils.utils'] = MagicMock()


# Add app to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.app_ui import AppUI
from app.config import Config

class TestAppStartup(unittest.TestCase):

    @patch('app.app_ui.UnifiedLogger')
    @patch('app.app_ui.ThumbnailManager')
    def test_app_ui_builds_successfully(self, mock_thumbnail_manager, mock_logger):
        """
        Test that the Gradio UI can be built without errors.
        This acts as a regression test for UI initialization issues.
        """
        try:
            # We need to mock the dependencies of AppUI
            config = Config()
            app_ui = AppUI(config=config, logger=mock_logger(), progress_queue=MagicMock(), cancel_event=MagicMock())

            # This is the critical part: building the UI
            # If there are any issues with component placement or arguments, this will fail
            demo = app_ui.build_ui()

            # We can also assert that the demo object is not None
            self.assertIsNotNone(demo, "The Gradio UI object should not be None.")

        except Exception as e:
            self.fail(f"Building the Gradio UI failed with an exception: {e}")

if __name__ == '__main__':
    unittest.main()