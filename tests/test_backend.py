import sys
from unittest.mock import MagicMock, patch

import pytest

from app.backend import Backend
from app.config import Config
from app.logging import UnifiedLogger
from app.thumb_cache import ThumbnailManager


@pytest.fixture
def backend():
    # Mock heavy dependencies
    sys.modules['torch'] = MagicMock()
    sys.modules['torch.cuda'] = MagicMock()
    sys.modules['torchvision'] = MagicMock()
    sys.modules['cv2'] = MagicMock()
    sys.modules['insightface'] = MagicMock()
    sys.modules['grounding_dino'] = MagicMock()
    sys.modules['ultralytics'] = MagicMock()

    config = Config()
    logger = UnifiedLogger()
    progress_queue = MagicMock()
    cancel_event = MagicMock()
    thumbnail_manager = ThumbnailManager(max_size=10)

    return Backend(
        config=config,
        logger=logger,
        progress_queue=progress_queue,
        cancel_event=cancel_event,
        thumbnail_manager=thumbnail_manager,
        cuda_available=False
    )


def test_backend_init(backend):
    assert backend is not None


@patch('app.backend.execute_extraction')
def test_run_extraction_wrapper(mock_execute_extraction, backend):
    mock_execute_extraction.return_value = iter([
        {
            "done": True,
            "log": "Extraction complete.",
            "extracted_video_path_state": "/path/to/video.mp4",
            "extracted_frames_dir_state": "/path/to/frames"
        }
    ])

    result = backend.run_extraction_wrapper(
        'test_video.mp4',
        None,
        'scene',
        '1',
        '10',
        True,
        '1080',
        False,
        True,
        0.5,
        True
    )

    assert result[0] == "Extraction complete."
    assert result[1] == "/path/to/video.mp4"
    assert result[2] == "/path/to/frames"


@patch('app.backend.execute_extraction')
def test_run_extraction_wrapper_failure(mock_execute_extraction, backend):
    mock_execute_extraction.return_value = iter([
        {
            "done": False,
            "log": "Extraction failed."
        }
    ])

    result = backend.run_extraction_wrapper(
        'test_video.mp4',
        None,
        'scene',
        '1',
        '10',
        True,
        '1080',
        False,
        True,
        0.5,
        True
    )

    assert result[0] == "Extraction failed."


@patch('app.backend.save_scene_seeds')
@patch('app.backend.execute_pre_analysis')
def test_run_pre_analysis_wrapper(mock_execute_pre_analysis, mock_save_scene_seeds, backend):
    mock_execute_pre_analysis.return_value = iter([
        {
            "done": True,
            "log": "Pre-analysis complete.",
            "previews": [("preview1", "caption1")],
            "scenes": [{"shot_id": 0, "status": "included"}],
            "output_dir": "/path/to/frames"
        }
    ])

    result = backend.run_pre_analysis_wrapper(
        'output_folder', 'video_path', False, False, 'face_ref_img_path', None,
        'face_model_name', True, 'dam4sam_model_name', 'person_detector_model',
        'seed_strategy', True, False, 'text_prompt', 0.5, 0.5, 5.0, 1.0, 1.0,
        'gdino_config_path', 'gdino_checkpoint_path', True, 1, '🤖 Automatic'
    )

    assert result[0] == "Pre-analysis complete."
    assert result[2] == [{"shot_id": 0, "status": "included"}]


@patch('app.backend.execute_pre_analysis')
def test_run_pre_analysis_wrapper_failure(mock_execute_pre_analysis, backend):
    mock_execute_pre_analysis.return_value = iter([
        {
            "done": False,
            "log": "Pre-analysis failed."
        }
    ])

    result = backend.run_pre_analysis_wrapper(
        'output_folder', 'video_path', False, False, 'face_ref_img_path', None,
        'face_model_name', True, 'dam4sam_model_name', 'person_detector_model',
        'seed_strategy', True, False, 'text_prompt', 0.5, 0.5, 5.0, 1.0, 1.0,
        'gdino_config_path', 'gdino_checkpoint_path', True, 1, '🤖 Automatic'
    )

    assert result[0] == "Pre-analysis failed."


@patch('app.backend.execute_propagation')
def test_run_propagation_wrapper(mock_execute_propagation, backend):
    mock_execute_propagation.return_value = iter([
        {
            "done": True,
            "log": "Propagation complete.",
            "output_dir": "/path/to/frames",
            "metadata_path": "/path/to/frames/metadata.json"
        }
    ])

    result = backend.run_propagation_wrapper(
        [{"shot_id": 0, "status": "included"}],
        'output_folder', 'video_path', False, False, 'face_ref_img_path', None,
        'face_model_name', True, 'dam4sam_model_name', 'person_detector_model',
        'seed_strategy', True, False, 'text_prompt', 0.5, 0.5, 5.0, 1.0, 1.0,
        'gdino_config_path', 'gdino_checkpoint_path', True, 1, '🤖 Automatic'
    )

    assert result[0] == "Propagation complete."
    assert result[1] == "/path/to/frames"
    assert result[2] == "/path/to/frames/metadata.json"


@patch('app.backend.execute_propagation')
def test_run_propagation_wrapper_failure(mock_execute_propagation, backend):
    mock_execute_propagation.return_value = iter([
        {
            "done": False,
            "log": "Propagation failed."
        }
    ])

    result = backend.run_propagation_wrapper(
        [{"shot_id": 0, "status": "included"}],
        'output_folder', 'video_path', False, False, 'face_ref_img_path', None,
        'face_model_name', True, 'dam4sam_model_name', 'person_detector_model',
        'seed_strategy', True, False, 'text_prompt', 0.5, 0.5, 5.0, 1.0, 1.0,
        'gdino_config_path', 'gdino_checkpoint_path', True, 1, '🤖 Automatic'
    )

    assert result[0] == "Propagation failed."


@patch('app.backend.execute_session_load')
def test_run_session_load_wrapper(mock_execute_session_load, backend):
    mock_execute_session_load.return_value = iter([
        {
            "unified_log": "Session loaded."
        }
    ])

    result = backend.run_session_load_wrapper(
        '/path/to/session'
    )

    # The wrapper is a generator, so we need to iterate over it
    result_list = list(result)

    assert result_list[0][0] == "Session loaded."