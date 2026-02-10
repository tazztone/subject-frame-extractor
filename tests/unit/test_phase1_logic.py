
import unittest
from unittest.mock import MagicMock, patch
from ui.app_ui import AppUI
from core.application_state import ApplicationState
import gradio as gr

class TestPhase1Logic(unittest.TestCase):
    def setUp(self):
        self.config = MagicMock()
        self.logger = MagicMock()
        self.progress_queue = MagicMock()
        self.cancel_event = MagicMock()
        self.thumbnail_manager = MagicMock()
        self.model_registry = MagicMock()
        
        with patch('ui.app_ui.torch.cuda.is_available', return_value=False):
            self.app = AppUI(
                self.config, self.logger, self.progress_queue, 
                self.cancel_event, self.thumbnail_manager, self.model_registry
            )
            # Mock components
            self.app.components = {
                "application_state": gr.State(),
                "seeding_results_column": gr.Column(),
                "propagation_group": gr.Group(),
                "propagate_masks_button": gr.Button(),
                "scene_filter_status": gr.Markdown(),
                "unified_status": gr.Markdown(),
                "unified_log": gr.Textbox(),
            }

    @patch('ui.app_ui.is_image_folder')
    @patch('ui.app_ui.get_scene_status_text')
    def test_on_pre_analysis_success_image_folder(self, mock_get_status, mock_is_image_folder):
        mock_is_image_folder.return_value = True
        mock_get_status.return_value = ("status", gr.update(interactive=True))
        
        result = {
            "scenes": [],
            "output_dir": "/mock/images"
        }
        current_state = ApplicationState()
        
        with patch('ui.app_ui.Scene', side_effect=lambda **kwargs: kwargs):
            updates = self.app._on_pre_analysis_success(result, current_state)
            
            # Check if propagate_masks_button is hidden
            button_update = updates[self.app.components["propagate_masks_button"]]
            self.assertFalse(button_update.get("visible", True))
            
            # Check success message
            status_msg = updates[self.app.components["unified_status"]]
            self.assertIn("compute metrics", status_msg)

    @patch('ui.app_ui.is_image_folder')
    @patch('ui.app_ui.get_scene_status_text')
    def test_on_pre_analysis_success_video_folder(self, mock_get_status, mock_is_image_folder):
        mock_is_image_folder.return_value = False
        mock_get_status.return_value = ("status", gr.update(interactive=True))
        
        result = {
            "scenes": [],
            "output_dir": "/mock/video"
        }
        current_state = ApplicationState(extracted_video_path="/mock/video.mp4")
        
        with patch('ui.app_ui.Scene', side_effect=lambda **kwargs: kwargs):
            updates = self.app._on_pre_analysis_success(result, current_state)
            
            # Check if propagate_masks_button is visible
            button_update = updates[self.app.components["propagate_masks_button"]]
            self.assertTrue(button_update.get("visible", False))
            
            # Check success message
            status_msg = updates[self.app.components["unified_status"]]
            self.assertIn("propagate masks", status_msg)

    def test_propagation_button_handler_guard(self):
        current_state = ApplicationState(extracted_video_path="") # Image mode
        
        gen = self.app._propagation_button_handler(current_state)
        updates = next(gen)
        
        self.assertIn("not needed for image folders", updates[self.app.components["unified_log"]])

if __name__ == '__main__':
    unittest.main()
