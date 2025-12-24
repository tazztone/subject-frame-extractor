
import pytest
import numpy as np
import cv2
import json
from unittest.mock import MagicMock, patch, ANY
from pathlib import Path
from core.scene_utils.helpers import (
    draw_boxes_preview,
    save_scene_seeds,
    get_scene_status_text,
    toggle_scene_status,
    _create_analysis_context,
    _recompute_single_preview,
    _wire_recompute_handler,
)
from core.models import Scene, AnalysisParameters

class TestSceneUtilsHelpers:

    @pytest.fixture
    def mock_scene(self):
        return Scene(
            shot_id=1,
            start_frame=0,
            end_frame=10,
            status="pending",
            best_frame=5,
            seed_frame_idx=5,
            seed_type="auto",
            seed_config={},
            seed_metrics={}
        )

    @pytest.fixture
    def mock_logger(self):
        return MagicMock()

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.visualization_bbox_color = (0, 255, 0)
        config.visualization_bbox_thickness = 2
        config.seeding_iou_threshold = 0.5
        config.seeding_face_contain_score = 10.0
        config.seeding_confidence_score_multiplier = 1.0
        config.seeding_iou_bonus = 5.0
        config.seeding_balanced_score_weights = {'area': 1.0, 'confidence': 1.0, 'edge': 1.0}
        config.seeding_face_to_body_expansion_factors = [1.5, 3.0, 1.0]
        config.seeding_final_fallback_box = [0.25, 0.25, 0.75, 0.75]
        return config

    def test_draw_boxes_preview(self, mock_config):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes = [[10, 10, 50, 50], [60, 60, 80, 80]]

        result = draw_boxes_preview(img, boxes, mock_config)

        assert result.shape == img.shape
        # Check if rectangles were drawn (some pixels should be green)
        # Note: BGR is used by cv2, but config.visualization_bbox_color is passed directly.
        # Assuming (0, 255, 0) is passed, channel 1 should be 255 at the box locations.
        assert np.any(result[10:50, 10:50, 1] == 255)
        assert np.any(result[60:80, 60:80, 1] == 255)

    def test_save_scene_seeds(self, mock_scene, mock_logger, tmp_path):
        scenes = [mock_scene]
        output_dir = tmp_path

        save_scene_seeds(scenes, str(output_dir), mock_logger)

        file_path = output_dir / "scene_seeds.json"
        assert file_path.exists()

        content = json.loads(file_path.read_text())
        assert str(mock_scene.shot_id) in content
        assert content[str(mock_scene.shot_id)]['best_frame'] == 5

    def test_save_scene_seeds_empty(self, mock_logger):
        save_scene_seeds([], "some/path", mock_logger)
        mock_logger.info.assert_not_called()

    def test_save_scene_seeds_error(self, mock_scene, mock_logger):
        # Passing an invalid path that raises error
        with patch('pathlib.Path.write_text', side_effect=Exception("Write failed")):
            save_scene_seeds([mock_scene], "some/path", mock_logger)
            mock_logger.error.assert_called_once()

    def test_get_scene_status_text(self, mock_scene):
        mock_scene.status = 'included'
        mock_scene.seed_result = {'bbox': [10, 10, 50, 50]}

        scenes = [mock_scene]
        status, button_update = get_scene_status_text(scenes)

        assert "1/1 scenes included" in status
        assert "Propagate Masks on 1" in button_update['value']
        assert button_update['interactive'] is True

    def test_get_scene_status_text_empty(self):
        status, button_update = get_scene_status_text([])
        assert "No scenes loaded" in status
        assert button_update['interactive'] is False

    def test_get_scene_status_text_rejected(self):
        scene = Scene(shot_id=1, start_frame=0, end_frame=10, status='excluded')
        scene.rejection_reasons = ['too_short']

        status, _ = get_scene_status_text([scene])
        assert "Rejected: too_short: 1" in status

    def test_toggle_scene_status(self, mock_scene, mock_logger, tmp_path):
        scenes = [mock_scene]
        new_status = 'excluded'

        updated_scenes, status_text, msg, btn = toggle_scene_status(
            scenes, mock_scene.shot_id, new_status, str(tmp_path), mock_logger
        )

        assert updated_scenes[0].status == new_status
        assert updated_scenes[0].manual_status_change is True
        assert f"set to {new_status}" in msg
        # Check if saved
        assert (tmp_path / "scene_seeds.json").exists()

    def test_toggle_scene_status_not_found(self, mock_scene, mock_logger, tmp_path):
        updated_scenes, status_text, msg, btn = toggle_scene_status(
            [mock_scene], 999, 'excluded', str(tmp_path), mock_logger
        )
        assert "Could not find scene" in msg

    @patch('core.scene_utils.helpers.initialize_analysis_models')
    @patch('core.scene_utils.helpers.create_frame_map')
    def test_create_analysis_context(self, mock_create_frame_map, mock_init_models, mock_config, mock_logger):
        # Mock inputs
        thumbnail_manager = MagicMock()
        model_registry = MagicMock()

        # Mock config values needed for AnalysisParameters
        mock_config.tracker_model_name = "sam3"
        mock_config.default_max_resolution = "1080"

        ana_ui_map_keys = ['output_folder', 'tracker_model_name']
        ana_input_components = ['/tmp/out', 'sam3']

        # Setup mocks
        mock_init_models.return_value = {
            "face_analyzer": MagicMock(),
            "ref_emb": None,
            "face_landmarker": MagicMock(),
            "device": "cpu"
        }
        mock_create_frame_map.return_value = {0: "frame_0.jpg"}

        with patch('pathlib.Path.exists', return_value=True), \
             patch('pathlib.Path.resolve', return_value=Path('/tmp/out')):

            masker = _create_analysis_context(
                mock_config, mock_logger, thumbnail_manager, False,
                ana_ui_map_keys, ana_input_components, model_registry
            )

            assert masker is not None
            assert masker.params.output_folder == str(Path('/tmp/out'))

    @patch('core.scene_utils.helpers.render_mask_overlay')
    @patch('PIL.Image.fromarray')
    def test_recompute_single_preview(self, mock_pil_fromarray, mock_render, mock_scene, mock_logger, mock_config):
        # Setup
        mock_scene.best_frame = 5
        scene_state = MagicMock()
        scene_state.scene = mock_scene

        masker = MagicMock()
        masker.params.output_folder = "/tmp/out"
        masker.frame_map = {5: "frame_5.webp"}
        masker.get_seed_for_frame.return_value = ([10, 10, 50, 50], {'final_score': 0.9})
        masker.get_mask_for_bbox.return_value = np.ones((100, 100), dtype=np.uint8)

        thumbnail_manager = MagicMock()
        thumbnail_manager.get.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        overrides = {'text_prompt': 'a person'}

        with patch('pathlib.Path.mkdir'), patch('pathlib.Path.exists', return_value=True):
            _recompute_single_preview(scene_state, masker, overrides, thumbnail_manager, mock_logger)

            # Verifications
            masker.get_seed_for_frame.assert_called_once()
            scene_state.update_seed_result.assert_called_once()
            assert mock_scene.seed_metrics['score'] == 0.9
            mock_pil_fromarray.return_value.save.assert_called_once()
            assert mock_scene.preview_path is not None

    @patch('core.scene_utils.helpers._create_analysis_context')
    @patch('core.scene_utils.helpers._recompute_single_preview')
    @patch('core.scene_utils.helpers.save_scene_seeds')
    @patch('core.scene_utils.helpers.build_scene_gallery_items')
    def test_wire_recompute_handler(self, mock_build_gallery, mock_save, mock_recompute, mock_create_context, mock_config, mock_logger, mock_scene):
        # Setup
        thumbnail_manager = MagicMock()
        model_registry = MagicMock()
        scenes = [mock_scene]
        shot_id = 1
        outdir = "/tmp/out"
        text_prompt = "a person"
        view = "grid"

        mock_build_gallery.return_value = ([], [], [])

        scenes, gal_update, idx_update, msg = _wire_recompute_handler(
            mock_config, mock_logger, thumbnail_manager, scenes, shot_id, outdir,
            text_prompt, view, [], [], False, model_registry
        )

        assert "successfully" in msg
        mock_create_context.assert_called_once()
        mock_recompute.assert_called_once()
        mock_save.assert_called_once()

    def test_wire_recompute_handler_no_prompt(self, mock_logger):
        res = _wire_recompute_handler(
            MagicMock(), mock_logger, MagicMock(), [], 1, "", "", "", [], [], False, MagicMock()
        )
        assert "Enter a text prompt" in res[3]
