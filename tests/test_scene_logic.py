import pytest
from unittest.mock import patch, MagicMock

from app.scene_logic import (
    get_scene_status_text,
    toggle_scene_status,
    apply_bulk_scene_filters,
)

# --- Test Data ---

@pytest.fixture
def sample_scenes():
    """Provides a list of sample scene dictionaries for testing."""
    return [
        {'shot_id': 1, 'status': 'pending', 'seed_result': {'details': {'mask_area_pct': 50}}, 'seed_metrics': {'best_face_sim': 0.9, 'score': 0.95}},
        {'shot_id': 2, 'status': 'pending', 'seed_result': {'details': {'mask_area_pct': 5}}, 'seed_metrics': {'best_face_sim': 0.8, 'score': 0.9}},
        {'shot_id': 3, 'status': 'pending', 'seed_result': {'details': {'mask_area_pct': 60}}, 'seed_metrics': {'best_face_sim': 0.4, 'score': 0.8}},
        {'shot_id': 4, 'status': 'pending', 'seed_result': {'details': {'mask_area_pct': 70}}, 'seed_metrics': {'score': 0.7}}, # No face sim
    ]

# --- Tests for get_scene_status_text ---

def test_get_scene_status_text_no_scenes():
    assert get_scene_status_text([]) == "No scenes loaded."

def test_get_scene_status_text_with_scenes():
    scenes = [{'status': 'included'}, {'status': 'excluded'}, {'status': 'included'}]
    assert get_scene_status_text(scenes) == "2/3 scenes included for propagation."

# --- Tests for toggle_scene_status ---

@patch('app.scene_logic.save_scene_seeds')
def test_toggle_scene_status(mock_save_seeds, sample_scenes):
    scenes, status_text, unified_status = toggle_scene_status(sample_scenes, 2, 'included', '/fake/dir', MagicMock())

    assert scenes[1]['status'] == 'included'
    assert scenes[1]['manual_status_change'] is True
    assert "1/4 scenes included" in status_text
    assert "Scene 2 status set to included" in unified_status
    mock_save_seeds.assert_called_once()

def test_toggle_scene_status_no_scene_selected(sample_scenes):
    scenes, status_text, unified_status = toggle_scene_status(sample_scenes, None, 'included', '/fake/dir', MagicMock())
    assert unified_status == "No scene selected."

# --- Tests for apply_bulk_scene_filters ---

@patch('app.scene_logic.save_scene_seeds')
def test_apply_bulk_filters_mask_area(mock_save, sample_scenes):
    scenes, _ = apply_bulk_scene_filters(sample_scenes, 10, 0, 0, False, '/fake/dir', MagicMock())
    assert scenes[0]['status'] == 'included'
    assert scenes[1]['status'] == 'excluded' # mask_area_pct is 5, less than 10
    assert scenes[2]['status'] == 'included'
    assert scenes[3]['status'] == 'included'

@patch('app.scene_logic.save_scene_seeds')
def test_apply_bulk_filters_face_sim(mock_save, sample_scenes):
    scenes, _ = apply_bulk_scene_filters(sample_scenes, 0, 0.5, 0, True, '/fake/dir', MagicMock())
    assert scenes[0]['status'] == 'included'
    assert scenes[1]['status'] == 'included'
    assert scenes[2]['status'] == 'excluded' # face_sim is 0.4, less than 0.5
    assert scenes[3]['status'] == 'included' # No face sim, so it is not excluded

@patch('app.scene_logic.save_scene_seeds')
def test_apply_bulk_filters_confidence(mock_save, sample_scenes):
    scenes, _ = apply_bulk_scene_filters(sample_scenes, 0, 0, 0.85, False, '/fake/dir', MagicMock())
    assert scenes[0]['status'] == 'included'
    assert scenes[1]['status'] == 'included'
    assert scenes[2]['status'] == 'excluded' # score is 0.8, less than 0.85
    assert scenes[3]['status'] == 'excluded' # score is 0.7, less than 0.85

@patch('app.scene_logic.save_scene_seeds')
def test_apply_bulk_filters_resets_manual_status(mock_save, sample_scenes):
    sample_scenes[0]['manual_status_change'] = True
    scenes, _ = apply_bulk_scene_filters(sample_scenes, 0, 0, 0, False, '/fake/dir', MagicMock())
    assert all(not s.get('manual_status_change', False) for s in scenes)