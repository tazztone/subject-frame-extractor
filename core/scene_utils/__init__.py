"""
Scene utilities package for Frame Extractor & Analyzer.

This package provides scene detection, seed selection, mask propagation,
and related utilities. All public symbols are re-exported for backward
compatibility.

Example usage:
    from core.scene_utils import SubjectMasker, run_scene_detection
    from core.scene_utils import MaskPropagator, SeedSelector
"""
from __future__ import annotations

# Detection functions
from core.scene_utils.detection import (
    run_scene_detection,
    make_photo_thumbs,
)

# Classes
from core.scene_utils.mask_propagator import MaskPropagator
from core.scene_utils.seed_selector import SeedSelector
from core.scene_utils.subject_masker import SubjectMasker

# Helper functions
from core.scene_utils.helpers import (
    draw_boxes_preview,
    save_scene_seeds,
    get_scene_status_text,
    toggle_scene_status,
    _create_analysis_context,
    _recompute_single_preview,
    _wire_recompute_handler,
)

__all__ = [
    # Detection
    'run_scene_detection',
    'make_photo_thumbs',
    # Classes
    'MaskPropagator',
    'SeedSelector',
    'SubjectMasker',
    # Helpers
    'draw_boxes_preview',
    'save_scene_seeds',
    'get_scene_status_text',
    'toggle_scene_status',
    '_create_analysis_context',
    '_recompute_single_preview',
    '_wire_recompute_handler',
]
