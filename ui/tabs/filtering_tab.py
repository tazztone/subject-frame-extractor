from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

if TYPE_CHECKING:
    from ui.app_ui import AppUI


class FilteringTabBuilder:
    def __init__(self, app: "AppUI"):
        self.app = app
        self.config = app.config

    def build(self):
        """Creates the content for the 'Export' tab."""
        self.app._create_section_header("Step 5: Filter & Export", "Fine-tune your dataset and save the best frames.")

        # 1. Global Filter Controls (Top Row)
        with gr.Group():
            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    self.app._create_component(
                        "filter_preset_dropdown",
                        "dropdown",
                        {
                            "label": "Use a Preset",
                            "choices": ["None"] + list(self.app.FILTER_PRESETS.keys()),
                            "info": "Apply standard settings for common use-cases.",
                        },
                    )
                with gr.Column(scale=3):
                    with gr.Row():
                        self.app._create_component(
                            "smart_filter_checkbox",
                            "checkbox",
                            {
                                "label": "Smart Filtering (Percentile)",
                                "value": False,
                                "info": "Keep top X% instead of absolute values.",
                            },
                        )
                        self.app._create_component(
                            "auto_pctl_input",
                            "slider",
                            {
                                "label": "Target %",
                                "minimum": 1,
                                "maximum": 99,
                                "value": self.config.gradio_auto_pctl_input,
                                "step": 1,
                                "container": False,
                            },
                        )
                with gr.Column(scale=1):
                    self.app._create_component(
                        "apply_auto_button",
                        "button",
                        {"value": "⚡ Auto-Threshold", "size": "sm", "variant": "secondary"},
                    )
                    self.app._create_component(
                        "reset_filters_button", "button", {"value": "🔄 Reset All", "size": "sm"}
                    )

        with gr.Row():
            # Left Column: Controls (Filters)
            with gr.Column(scale=1, min_width=400):
                self.app._create_component("filter_status_text", "markdown", {"value": "*Analysis not loaded.*"})

                # Dynamic Component Registry
                self.app.components["metric_plots"] = {}
                self.app.components["metric_sliders"] = {}
                self.app.components["metric_accs"] = {}
                self.app.components["metric_auto_threshold_cbs"] = {}

                # 2. Deduplication Accordion
                with gr.Accordion("👯 Deduplication (Remove Duplicates)", open=True) as dedup_acc:
                    self.app.components["metric_accs"]["dedup"] = dedup_acc
                    self.app._create_component(
                        "dedup_method_input",
                        "dropdown",
                        {
                            "label": "Method",
                            "choices": ["Off", "Fast (pHash)", "Accurate (LPIPS)"],
                            "value": "Fast (pHash)",
                        },
                    )

                    self.app._create_component(
                        "dedup_thresh_input",
                        "slider",
                        {
                            "label": "Sensitivity Threshold",
                            "minimum": -1,
                            "maximum": 32,
                            "value": 5,
                            "step": 1,
                            "info": "Lower = stricter (keeps more similar frames), Higher = removes more.",
                        },
                    )
                    # Hidden inputs
                    self.app._create_component("ssim_threshold_input", "slider", {"visible": False, "value": 0.95})
                    self.app._create_component("lpips_threshold_input", "slider", {"visible": False, "value": 0.1})

                    with gr.Row():
                        self.app._create_component(
                            "dedup_visual_diff_input",
                            "checkbox",
                            {"label": "Show Diff", "value": False, "visible": False},
                        )
                        self.app._create_component(
                            "calculate_diff_button", "button", {"value": "🔍 Inspect Duplicates"}
                        )
                    self.app._create_component("visual_diff_image", "image", {"label": "Visual Diff", "visible": False})

                # 3. Dynamic Metric Accordions
                metric_configs = {
                    "quality_score": {"open": True},
                    "niqe": {"open": False},
                    "sharpness": {"open": False},
                    "edge_strength": {"open": False},
                    "contrast": {"open": False},
                    "brightness": {"open": False},
                    "entropy": {"open": False},
                    "face_sim": {"open": False},
                    "mask_area_pct": {"open": False},
                    "eyes_open": {"open": False},
                    "yaw": {"open": False},
                    "pitch": {"open": False},
                }
                for metric_name, metric_config in metric_configs.items():
                    if not hasattr(self.config, f"filter_default_{metric_name}"):
                        continue
                    f_def = getattr(self.config, f"filter_default_{metric_name}")

                    with gr.Accordion(
                        metric_name.replace("_", " ").title(), open=metric_config["open"], visible=False
                    ) as acc:
                        self.app.components["metric_accs"][metric_name] = acc
                        gr.Markdown(self.app.get_metric_description(metric_name), elem_classes="metric-description")
                        with gr.Column(elem_classes="plot-and-slider-column"):
                            self.app.components["metric_plots"][metric_name] = self.app._create_component(
                                f"plot_{metric_name}", "html", {"visible": True}
                            )

                            with gr.Row():
                                self.app.components["metric_sliders"][f"{metric_name}_min"] = (
                                    self.app._create_component(
                                        f"slider_{metric_name}_min",
                                        "slider",
                                        {
                                            "label": "Min",
                                            "minimum": f_def["min"],
                                            "maximum": f_def["max"],
                                            "value": f_def.get("default_min", f_def["min"]),
                                            "step": f_def["step"],
                                            "interactive": True,
                                            "visible": True,
                                        },
                                    )
                                )
                                if "default_max" in f_def:
                                    self.app.components["metric_sliders"][f"{metric_name}_max"] = (
                                        self.app._create_component(
                                            f"slider_{metric_name}_max",
                                            "slider",
                                            {
                                                "label": "Max",
                                                "minimum": f_def["min"],
                                                "maximum": f_def["max"],
                                                "value": f_def["default_max"],
                                                "step": f_def["step"],
                                                "interactive": True,
                                                "visible": True,
                                            },
                                        )
                                    )

                            self.app.components["metric_auto_threshold_cbs"][metric_name] = self.app._create_component(
                                f"auto_threshold_{metric_name}",
                                "checkbox",
                                {"label": "Auto-Threshold", "value": False, "interactive": True, "visible": True},
                            )
                            if metric_name == "face_sim":
                                self.app._create_component(
                                    "require_face_match_input",
                                    "checkbox",
                                    {
                                        "label": "Reject if no face",
                                        "value": self.config.default_require_face_match,
                                        "visible": True,
                                    },
                                )

            # Right Column: Results & Export
            with gr.Column(scale=2):
                with gr.Group(visible=False) as results_group:
                    self.app.components["results_group"] = results_group
                    gr.Markdown("#### 🖼️ Results Preview")
                    with gr.Row():
                        self.app._create_component(
                            "gallery_view_toggle",
                            "radio",
                            {
                                "choices": self.app.GALLERY_VIEW_CHOICES,
                                "value": "Kept",
                                "label": "Show",
                                "container": False,
                            },
                        )
                        self.app._create_component(
                            "show_mask_overlay_input",
                            "checkbox",
                            {"label": "Mask Overlay", "value": self.config.gradio_show_mask_overlay},
                        )
                        self.app._create_component(
                            "overlay_alpha_slider",
                            "slider",
                            {
                                "label": "Alpha",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "value": self.config.gradio_overlay_alpha,
                                "step": 0.1,
                                "container": False,
                            },
                        )
                    self.app._create_component(
                        "results_gallery",
                        "gallery",
                        {
                            "columns": [4, 6, 8],
                            "rows": 2,
                            "height": "auto",
                            "preview": True,
                            "allow_preview": True,
                            "object_fit": "contain",
                            "show_label": False,
                        },
                    )

                with gr.Group(visible=False) as export_group:
                    self.app.components["export_group"] = export_group
                    gr.Markdown("#### 📤 Export Dataset")

                    with gr.Accordion("Advanced Export Options", open=False):
                        with gr.Row():
                            self.app._create_component(
                                "enable_crop_input",
                                "checkbox",
                                {"label": "✂️ Crop to Subject", "value": self.config.export_enable_crop},
                            )
                            self.app._create_component(
                                "enable_xmp_export_input",
                                "checkbox",
                                {"label": "📝 Write XMP Sidecars (Photos)", "value": False},
                            )

                        with gr.Row():
                            self.app._create_component(
                                "crop_padding_input",
                                "slider",
                                {"label": "Crop Padding %", "value": self.config.export_crop_padding},
                            )
                        self.app._create_component(
                            "crop_ar_input",
                            "textbox",
                            {
                                "label": "Aspect Ratio (e.g., 1:1, 9:16)",
                                "value": self.config.export_crop_ars,
                                "info": "Force crops to specific aspect ratios.",
                            },
                        )

                    with gr.Row():
                        self.app._create_component(
                            "export_button",
                            "button",
                            {"value": "💾 Export Kept Frames", "variant": "primary", "scale": 2, "size": "lg"},
                        )
                        self.app._create_component(
                            "dry_run_button", "button", {"value": "Dry Run", "scale": 1, "size": "lg"}
                        )