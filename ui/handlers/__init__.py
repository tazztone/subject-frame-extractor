"""
UI Handlers package for Frame Extractor.

This package contains handler modules that encapsulate related UI functionality,
extracted from the monolithic AppUI class to improve maintainability.
"""

from __future__ import annotations

from ui.handlers.filtering_handler import FilteringHandler
from ui.handlers.pipeline_handlers import PipelineHandler
from ui.handlers.scene_handler import SceneHandler
from ui.handlers.subject_handler import SubjectHandler

__all__ = [
    "SceneHandler",
    "PipelineHandler",
    "SubjectHandler",
    "FilteringHandler",
]
