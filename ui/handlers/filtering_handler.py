from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import gradio as gr
import numpy as np

from core.application_state import ApplicationState
from core.events import FilterEvent
from ui.decorators import safe_ui_callback
from ui.gallery_utils import auto_set_thresholds, on_filters_changed

if TYPE_CHECKING:
    from ui.app_ui import AppUI


class FilteringHandler:
    """
    Handles filtering logic, auto-thresholds, presets, and updates to the results gallery.
    Extracted from monolith app_ui.py to improve maintainability.
    """

    def __init__(self, app: "AppUI"):
        self.app = app
        self.config = app.config
        self.logger = app.logger
        self.thumbnail_manager = app.thumbnail_manager

    @safe_ui_callback("Preset Change")
    def on_preset_changed(self, preset_name: str) -> list[Any]:
        """Updates filter sliders when a preset is selected."""
        is_preset_active = preset_name != "None" and preset_name in self.app.FILTER_PRESETS
        final_updates = []
        slider_keys = sorted(self.app.components["metric_sliders"].keys())
        preset_values = self.app.FILTER_PRESETS.get(preset_name, {})
        for key in slider_keys:
            if is_preset_active and key in preset_values:
                val = preset_values[key]
            else:
                metric_key = re.sub(r"_(min|max)$", "", key)
                default_key = "default_max" if key.endswith("_max") else "default_min"
                val = getattr(self.config, f"filter_default_{metric_key}", {}).get(default_key, 0)

            # If Preset, enable smart mode (0-100) except angles
            if is_preset_active and "yaw" not in key and "pitch" not in key:
                final_updates.append(
                    gr.update(
                        minimum=0.0,
                        maximum=100.0,
                        step=1.0,
                        value=val,
                        label=f"{self.app.components['metric_sliders'][key].label.split('(')[0].strip()} (%)",
                    )
                )
            elif "yaw" in key or "pitch" in key:
                final_updates.append(gr.update(value=val))
            else:
                f_def = getattr(self.config, f"filter_default_{re.sub(r'_(min|max)$', '', key)}", {})
                final_updates.append(
                    gr.update(
                        minimum=f_def.get("min", 0),
                        maximum=f_def.get("max", 100),
                        step=f_def.get("step", 0.5),
                        value=val,
                        label=self.app.components["metric_sliders"][key].label.replace(" (%)", ""),
                    )
                )

        return [gr.update()] + final_updates + [gr.update(value=is_preset_active)]

    @safe_ui_callback("Filter Change")
    def on_filters_changed_wrapper(
        self,
        state: ApplicationState,
        gallery_view: str,
        show_overlay: bool,
        overlay_alpha: float,
        require_face_match: bool,
        dedup_thresh: int,
        dedup_method_ui: str,
        *slider_values: float,
    ) -> tuple[str, Any]:
        """
        Updates the results gallery when filters change.
        """
        slider_keys = sorted(self.app.components["metric_sliders"].keys())
        assert len(slider_values) == len(slider_keys), (
            f"Expected {len(slider_keys)} slider values, got {len(slider_values)}"
        )

        all_frames_data = state.all_frames_data
        per_metric_values = state.per_metric_values
        output_dir = state.analysis_output_dir
        smart_mode_enabled = state.smart_filter_enabled

        slider_values_dict = {k: v for k, v in zip(slider_keys, slider_values)}
        if smart_mode_enabled and per_metric_values:
            for key, val in slider_values_dict.items():
                if "yaw" in key or "pitch" in key:
                    continue
                metric_data = per_metric_values.get(re.sub(r"_(min|max)$", "", key))
                if metric_data:
                    try:
                        slider_values_dict[key] = float(np.percentile(np.array(metric_data), val))
                    except Exception:
                        pass

        dedup_method = self.app._map_dedup_method(dedup_method_ui)
        result = on_filters_changed(
            FilterEvent(
                all_frames_data=all_frames_data,
                per_metric_values=per_metric_values,
                output_dir=output_dir,
                gallery_view=gallery_view,
                show_overlay=show_overlay,
                overlay_alpha=overlay_alpha,
                require_face_match=require_face_match,
                dedup_thresh=dedup_thresh,
                slider_values=slider_values_dict,
                dedup_method=dedup_method,
            ),
            self.thumbnail_manager,
            self.config,
            self.logger,
        )
        return result["filter_status_text"], result["results_gallery"]

    @safe_ui_callback("Reset Filters")
    def on_reset_filters(self, state: ApplicationState) -> tuple:
        """Resets all filter settings to their defaults."""
        c = self.app.components
        slider_keys = sorted(c["metric_sliders"].keys())
        slider_updates = []
        for key in slider_keys:
            metric_key = re.sub(r"_(min|max)$", "", key)
            default_key = "default_max" if key.endswith("_max") else "default_min"
            val = getattr(self.config, f"filter_default_{metric_key}", {}).get(default_key, 0)
            slider_updates.append(gr.update(value=val))

        acc_updates = []
        for key in sorted(c["metric_accs"].keys()):
            acc_updates.append(gr.update(open=(key == "quality_score")))

        new_state = state.model_copy()
        new_state.smart_filter_enabled = False

        return tuple(
            [new_state]
            + slider_updates
            + [5, False, "Filters Reset.", gr.update(), "Fast (pHash)"]
            + acc_updates
            + [False]
        )

    @safe_ui_callback("Auto Thresholds")
    def on_auto_set_thresholds(self, per_metric_values: dict, p: int, *checkbox_values: bool) -> list[Any]:
        """Automatically sets filter thresholds based on data percentiles."""
        slider_keys = sorted(self.app.components["metric_sliders"].keys())
        auto_threshold_cbs_keys = sorted(self.app.components["metric_auto_threshold_cbs"].keys())

        assert len(checkbox_values) == len(auto_threshold_cbs_keys), (
            f"Expected {len(auto_threshold_cbs_keys)} checkbox values, got {len(checkbox_values)}"
        )

        selected_metrics = [
            metric_name for metric_name, is_selected in zip(auto_threshold_cbs_keys, checkbox_values) if is_selected
        ]
        updates = auto_set_thresholds(per_metric_values, p, slider_keys, selected_metrics)
        return [updates.get(f"slider_{key}", gr.update()) for key in slider_keys]
