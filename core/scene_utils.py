"""
Legacy scene_utils module.
This module is deprecated. Use core.scene_utils_pkg instead.
"""
from core.scene_utils_pkg import (
    run_scene_detection,
    make_photo_thumbs,
    MaskPropagator,
    SeedSelector,
    SubjectMasker,
    draw_boxes_preview,
    save_scene_seeds,
    get_scene_status_text,
    toggle_scene_status,
    _create_analysis_context,
    _recompute_single_preview,
    _wire_recompute_handler,
)

__all__ = [
    'run_scene_detection',
    'make_photo_thumbs',
    'MaskPropagator',
    'SeedSelector',
    'SubjectMasker',
    'draw_boxes_preview',
    'save_scene_seeds',
    'get_scene_status_text',
    'toggle_scene_status',
    '_create_analysis_context',
    '_recompute_single_preview',
    '_wire_recompute_handler',
]
