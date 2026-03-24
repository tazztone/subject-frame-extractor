"""
Extended tests for scene helpers to increase coverage.
"""

from unittest.mock import MagicMock

import pytest

from core.scene_utils.helpers import _create_analysis_context, _recompute_single_preview, toggle_scene_status


def test_toggle_scene_status_no_scene():
    logger = MagicMock()
    # No shot_id selected (Line 138-140)
    res = toggle_scene_status([], None, "included", "out", logger)
    assert res[2] == "No scene selected."

    # Scene not in list (Line 151)
    res = toggle_scene_status([MagicMock(shot_id=1)], 2, "included", "out", logger)
    assert res[2] == "Could not find scene 2."


def test_create_analysis_context_invalid_folder():
    config = MagicMock()
    logger = MagicMock()
    tm = MagicMock()
    mr = MagicMock()

    # Path is valid string but doesn't exist (Line 176-177)
    with pytest.raises(FileNotFoundError, match="Output folder is not valid"):
        _create_analysis_context(config, logger, tm, False, ["outputfolder"], ["/non/existent/path"], mr)


def test_recompute_single_preview_errors():
    scene_state = MagicMock()
    scene_state.scene.shot_id = 1
    scene_state.scene.best_frame = 5

    masker = MagicMock()
    masker.frame_map = {5: "frame_00005.jpg"}
    masker.params.output_folder = "out"

    tm = MagicMock()
    logger = MagicMock()

    # Thumbnail not found (Line 223)
    tm.get.return_value = None
    with pytest.raises(FileNotFoundError, match="Thumbnail for frame 5 not found"):
        _recompute_single_preview(scene_state, masker, {}, tm, logger)
