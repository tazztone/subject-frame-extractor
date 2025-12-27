"""
UI Handlers package for Frame Extractor.

This package contains handler modules that encapsulate related UI functionality,
extracted from the monolithic AppUI class to improve maintainability.
"""

from __future__ import annotations

from ui.handlers.analysis_handler import AnalysisHandler
from ui.handlers.extraction_handler import ExtractionHandler
from ui.handlers.filtering_handler import FilteringHandler

__all__ = [
    "ExtractionHandler",
    "AnalysisHandler",
    "FilteringHandler",
]
