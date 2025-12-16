"""
Extraction handler for Frame Extractor UI.

This module contains handlers related to the extraction pipeline,
including video extraction, session loading, and batch processing.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, Optional, Any, Generator
import gradio as gr

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger
    from core.managers import ThumbnailManager, ModelRegistry
    from core.events import ExtractionEvent, SessionLoadEvent
    from ui.app_ui import AppUI


class ExtractionHandler:
    """
    Handles extraction-related UI operations.
    
    Extracted from AppUI to reduce class size and improve maintainability.
    """
    
    def __init__(
        self,
        app: 'AppUI',
        config: 'Config',
        logger: 'AppLogger',
        thumbnail_manager: 'ThumbnailManager',
        model_registry: 'ModelRegistry'
    ):
        """
        Initialize ExtractionHandler.
        
        Args:
            app: Parent AppUI instance
            config: Application configuration
            logger: Application logger
            thumbnail_manager: Thumbnail cache manager
            model_registry: Model registry
        """
        self.app = app
        self.config = config
        self.logger = logger
        self.thumbnail_manager = thumbnail_manager
        self.model_registry = model_registry

    def run_extraction(
        self,
        progress: Callable,
        *args
    ) -> Generator[dict, None, None]:
        """
        Run the extraction pipeline.
        
        Args:
            progress: Gradio progress callback
            *args: UI component values
            
        Yields:
            Progress updates and final results
        """
        from core.events import ExtractionEvent
        from core.pipelines import execute_extraction
        
        event = ExtractionEvent.from_ui_args(self.app.ext_ui_map_keys, args)
        yield from execute_extraction(
            event,
            self.app.progress_queue,
            self.app.cancel_event,
            self.logger,
            self.config,
            self.thumbnail_manager,
            progress=progress,
            model_registry=self.model_registry
        )

    def on_extraction_success(self, result: dict) -> dict:
        """
        Handle successful extraction completion.
        
        Args:
            result: Result dictionary from extraction
            
        Returns:
            Dictionary of component updates
        """
        updates = {
            "unified_log": result.get("unified_log", "Extraction complete."),
            "extracted_video_path_state": result.get("extracted_video_path_state", ""),
            "extracted_frames_dir_state": result.get("extracted_frames_dir_state", ""),
            "main_tabs": gr.update(selected=1),  # Go to Define Subject tab
        }
        return updates

    def run_session_load(
        self,
        session_path: str
    ) -> Generator[dict, None, None]:
        """
        Load a previous session.
        
        Args:
            session_path: Path to session directory
            
        Yields:
            Progress updates and component updates
        """
        from core.events import SessionLoadEvent
        from core.pipelines import execute_session_load
        
        event = SessionLoadEvent(session_path=session_path)
        yield from execute_session_load(
            self.app,
            event,
            self.logger,
            self.config,
            self.thumbnail_manager,
            self.model_registry
        )
