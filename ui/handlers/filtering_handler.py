"""
Filtering handler for Frame Extractor UI.

This module contains handlers related to frame filtering and export.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger
    from core.managers import ThumbnailManager
    from ui.app_ui import AppUI


class FilteringHandler:
    """
    Handles filtering-related UI operations.

    Extracted from AppUI to reduce class size and improve maintainability.
    """

    def __init__(self, app: "AppUI", config: "Config", logger: "AppLogger", thumbnail_manager: "ThumbnailManager"):
        """
        Initialize FilteringHandler.

        Args:
            app: Parent AppUI instance
            config: Application configuration
            logger: Application logger
            thumbnail_manager: Thumbnail cache manager
        """
        self.app = app
        self.config = config
        self.logger = logger
        self.thumbnail_manager = thumbnail_manager

    def on_filters_changed(
        self,
        all_frames_data: list,
        per_metric_values: dict,
        output_dir: str,
        gallery_view: str,
        show_overlay: bool,
        overlay_alpha: float,
        require_face_match: bool,
        dedup_thresh: int,
        dedup_method_ui: str,
        smart_mode_enabled: bool,
        *slider_values: float,
    ) -> dict:
        """
        Handle filter changes and update gallery.

        Args:
            all_frames_data: All frame metadata
            per_metric_values: Per-metric value distributions
            output_dir: Output directory path
            gallery_view: Current gallery view mode
            show_overlay: Whether to show mask overlay
            overlay_alpha: Overlay transparency
            require_face_match: Whether face matching is required
            dedup_thresh: Deduplication threshold
            dedup_method_ui: Deduplication method
            smart_mode_enabled: Whether smart mode is enabled
            *slider_values: Slider values for each metric

        Returns:
            Dictionary of component updates
        """
        from core.events import FilterEvent
        from ui.gallery_utils import on_filters_changed

        slider_keys = self.app.get_all_filter_keys()
        slider_dict = dict(zip(slider_keys, slider_values))

        event = FilterEvent(
            all_frames_data=all_frames_data,
            per_metric_values=per_metric_values or {},
            slider_values=slider_dict,
            require_face_match=require_face_match,
            dedup_thresh=dedup_thresh,
            dedup_method=dedup_method_ui,
            output_dir=output_dir,
            gallery_view=gallery_view,
            show_overlay=show_overlay,
            overlay_alpha=overlay_alpha,
        )

        return on_filters_changed(event, self.thumbnail_manager, self.config, self.logger)

    def on_preset_changed(self, preset_name: str) -> dict:
        """
        Handle preset selection.

        Args:
            preset_name: Name of the selected preset

        Returns:
            Dictionary of slider updates
        """
        presets = {
            "No Filters": {},
            "Quality Focus": {"sharpness_min": 20, "contrast_min": 20, "edge_strength_min": 15},
            "Face Priority": {"face_sim_min": 0.6, "eyes_open_min": 0.5},
            "Balanced": {"sharpness_min": 15, "contrast_min": 15, "face_sim_min": 0.4},
        }
        preset = presets.get(preset_name, {})
        updates = {}
        for key in self.app.get_all_filter_keys():
            updates[f"slider_{key}"] = gr.update(value=preset.get(key, 0))
        return updates

    def on_reset_filters(self, all_frames_data: list, per_metric_values: dict, output_dir: str) -> dict:
        """
        Reset all filters to default values.

        Args:
            all_frames_data: All frame metadata
            per_metric_values: Per-metric value distributions
            output_dir: Output directory path

        Returns:
            Dictionary of component updates
        """
        updates = {}
        for key in self.app.get_all_filter_keys():
            updates[f"slider_{key}"] = gr.update(value=0)

        # Trigger filter update with reset values
        filter_result = self.on_filters_changed(
            all_frames_data,
            per_metric_values,
            output_dir,
            "Kept",
            False,
            0.6,
            False,
            5,
            "pHash",
            False,
            *([0] * len(self.app.get_all_filter_keys())),
        )
        updates.update(filter_result)
        return updates

    def on_auto_set_thresholds(self, per_metric_values: dict, percentile: int, *checkbox_values: bool) -> dict:
        """
        Automatically set thresholds based on percentiles.

        Args:
            per_metric_values: Per-metric value distributions
            percentile: Percentile to use for threshold
            *checkbox_values: Which metrics are enabled

        Returns:
            Dictionary of slider updates
        """
        from ui.gallery_utils import auto_set_thresholds

        slider_keys = self.app.get_all_filter_keys()
        metric_names = [k.replace("_min", "").replace("_max", "") for k in slider_keys]
        selected_metrics = [name for name, enabled in zip(metric_names, checkbox_values) if enabled]

        return auto_set_thresholds(per_metric_values or {}, percentile, slider_keys, selected_metrics)
