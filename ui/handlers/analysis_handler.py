"""
Analysis handler for Frame Extractor UI.

This module contains handlers related to the analysis pipelines,
including pre-analysis, propagation, and full analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Generator

import gradio as gr

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger
    from core.managers import ModelRegistry, ThumbnailManager
    from ui.app_ui import AppUI


class AnalysisHandler:
    """
    Handles analysis-related UI operations.

    Extracted from AppUI to reduce class size and improve maintainability.
    """

    def __init__(
        self,
        app: "AppUI",
        config: "Config",
        logger: "AppLogger",
        thumbnail_manager: "ThumbnailManager",
        model_registry: "ModelRegistry",
    ):
        """
        Initialize AnalysisHandler.

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

    def run_pre_analysis(self, progress: Callable, *args) -> Generator[dict, None, None]:
        """
        Run the pre-analysis pipeline.

        Args:
            progress: Gradio progress callback
            *args: UI component values

        Yields:
            Progress updates and final results
        """
        import torch

        from core.pipelines import execute_pre_analysis

        event = self.app._create_pre_analysis_event(*args)
        yield from execute_pre_analysis(
            event,
            self.app.progress_queue,
            self.app.cancel_event,
            self.logger,
            self.config,
            self.thumbnail_manager,
            cuda_available=torch.cuda.is_available(),
            progress=progress,
            model_registry=self.model_registry,
        )

    def on_pre_analysis_success(self, result: dict) -> dict:
        """
        Handle successful pre-analysis completion.

        Args:
            result: Result dictionary from pre-analysis

        Returns:
            Dictionary of component updates
        """
        updates = {
            "unified_log": result.get("unified_log", "Pre-analysis complete."),
            "scenes_state": result.get("scenes", []),
            "analysis_output_dir_state": result.get("output_dir", ""),
            "seeding_results_column": result.get("seeding_results_column", gr.update()),
            "propagation_group": result.get("propagation_group", gr.update()),
            "main_tabs": gr.update(selected=2),  # Go to Scene Selection tab
        }
        if result.get("final_face_ref_path"):
            updates["final_face_ref_path_state"] = result["final_face_ref_path"]
        return updates

    def run_propagation(self, scenes: list, progress: Callable, *args) -> Generator[dict, None, None]:
        """
        Run the mask propagation pipeline.

        Args:
            scenes: List of scenes to process
            progress: Gradio progress callback
            *args: UI component values

        Yields:
            Progress updates and final results
        """
        import torch

        from core.events import PropagationEvent
        from core.pipelines import execute_propagation

        event = PropagationEvent.from_ui_args(self.app.ana_ui_map_keys, scenes, args)
        yield from execute_propagation(
            event,
            self.app.progress_queue,
            self.app.cancel_event,
            self.logger,
            self.config,
            self.thumbnail_manager,
            cuda_available=torch.cuda.is_available(),
            progress=progress,
            model_registry=self.model_registry,
        )

    def on_propagation_success(self, result: dict) -> dict:
        """
        Handle successful propagation completion.

        Args:
            result: Result dictionary from propagation

        Returns:
            Dictionary of component updates
        """
        return {
            "unified_log": result.get("unified_log", "Propagation complete."),
            "analysis_output_dir_state": result.get("output_dir", ""),
            "filtering_tab": gr.update(interactive=True),
            "main_tabs": gr.update(selected=3),  # Go to Metrics tab
        }

    def run_analysis(self, scenes: list, progress: Callable, *args) -> Generator[dict, None, None]:
        """
        Run the full analysis pipeline.

        Args:
            scenes: List of scenes to process
            progress: Gradio progress callback
            *args: UI component values

        Yields:
            Progress updates and final results
        """
        import torch

        from core.events import PropagationEvent
        from core.pipelines import execute_analysis

        event = PropagationEvent.from_ui_args(self.app.ana_ui_map_keys, scenes, args)
        yield from execute_analysis(
            event,
            self.app.progress_queue,
            self.app.cancel_event,
            self.logger,
            self.config,
            self.thumbnail_manager,
            cuda_available=torch.cuda.is_available(),
            progress=progress,
            model_registry=self.model_registry,
        )

    def on_analysis_success(self, result: dict) -> dict:
        """
        Handle successful analysis completion.

        Args:
            result: Result dictionary from analysis

        Returns:
            Dictionary of component updates
        """
        return {
            "unified_log": result.get("unified_log", "Analysis complete."),
            "analysis_output_dir_state": result.get("output_dir", ""),
            "filtering_tab": gr.update(interactive=True),
            "main_tabs": gr.update(selected=4),  # Go to Filtering tab
        }
