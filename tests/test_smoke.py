# tests/test_smoke.py
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import app

def test_app_smoke_instantiates(tmp_path):
    # Patch CompositionRoot to avoid touching real models / files
    with patch("app.CompositionRoot") as MockRoot:
        mock_root = MockRoot.return_value
        mock_root.get_config.return_value = app.Config()
        mock_root.get_logger.return_value = MagicMock()

        ui = app.EnhancedAppUI()
        assert ui is not None

@patch('app.VideoManager')
def test_pipeline_smoke_runs_noop(MockVideoManager, tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    params_dict = {
        'source_path': 'test.mp4', 'video_path': '/dev/null', 'output_folder': str(out_dir),
        'method': 'interval', 'interval': '1.0', 'max_resolution': "720", 'thumbnails_only': True,
        'scene_detect': False, 'pre_analysis_enabled': False, 'enable_subject_mask': False,
        'primary_seed_strategy': '🧑‍🤝‍🧑 Find Prominent Person', 'person_detector_model': 'yolo11x.pt',
        'upload_video': None, 'nth_frame': '5', 'thumb_megapixels': 0.2, 'resume': False,
        'enable_face_filter': False, 'face_ref_img_path': '', 'face_ref_img_upload': None,
        'face_model_name': 'buffalo_l', 'dam4sam_model_name': 'sam21pp-T',
        'best_frame_strategy': 'Largest Person', 'text_prompt': '', 'box_threshold': 0.35,
        'text_threshold': 0.25, 'min_mask_area_pct': 1.0, 'sharpness_base_scale': 2500.0,
        'edge_strength_base_scale': 100.0, 'gdino_config_path': 'cfg.py',
        'gdino_checkpoint_path': 'ckpt.pth', 'pre_sample_nth': 1
    }
    params = app.ExtractionEvent.model_validate(params_dict)

    # Configure the mock VideoManager
    mock_video_manager_instance = MockVideoManager.return_value
    mock_video_manager_instance.prepare_video.return_value = "/dev/null"
    mock_video_manager_instance.get_video_info.return_value = (640, 360, 30.0, 10)


    with patch("app.EnhancedExtractionPipeline._run_ffmpeg"), \
         patch("app.validate_video_file", return_value=True):
        pipe = app.EnhancedExtractionPipeline(
            config=app.Config(),
            logger=MagicMock(),
            params=params,
            progress_queue=MagicMock(),
            cancel_event=MagicMock(),
        )

        # If your refactor moved model loading into an initializer, stub it:
        if hasattr(pipe, "_init_models"):
            pipe._init_models = MagicMock()

        # Should not raise
        pipe.run()
