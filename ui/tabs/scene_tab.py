from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

if TYPE_CHECKING:
    from ui.app_ui import AppUI


class SceneTabBuilder:
    def __init__(self, app: "AppUI"):
        self.app = app
        self.config = app.config

    def build(self):
        """Creates the content for the 'Scenes' tab."""
        self.app._create_section_header("Step 3: Scene Review", "Review detected scenes and confirm subject tracking.")

        with gr.Column(scale=2, visible=True) as seeding_results_column:
            self.app.components["seeding_results_column"] = seeding_results_column

            # 1. Editor Panel (Top, for easier access when selected)
            with gr.Group(visible=False, elem_classes="scene-editor") as scene_editor_group:
                self.app.components["scene_editor_group"] = scene_editor_group
                gr.Markdown("#### ✏️ Scene Editor")
                with gr.Row():
                    # Left: Preview
                    with gr.Column(scale=3):
                        self.app._create_component(
                            "gallery_image_preview",
                            "image",
                            {"label": "Shot Preview", "interactive": False, "height": 350, "show_label": False},
                        )

                    # Right: Controls
                    with gr.Column(scale=2):
                        self.app._create_component("sceneeditorstatusmd", "markdown", {"value": "**Selected Scene**"})

                        # Mini gallery for changing person ID
                        gr.Markdown("**Switch Subject:**")
                        self.app._create_component(
                            "subject_selection_gallery",
                            "gallery",
                            {
                                "label": "Detected People",
                                "columns": 4,
                                "height": "auto",
                                "allow_preview": False,
                                "object_fit": "cover",
                                "show_label": False,
                            },
                        )

                        with gr.Row():
                            self.app._create_component(
                                "sceneincludebutton",
                                "button",
                                {"value": "✅ Include", "variant": "secondary", "scale": 1},
                            )
                            self.app._create_component(
                                "sceneexcludebutton", "button", {"value": "❌ Exclude", "variant": "stop", "scale": 1}
                            )
                            self.app._create_component("sceneresetbutton", "button", {"value": "🔄 Reset", "scale": 1})

                        with gr.Accordion("🛠️ Manual Override", open=False):
                            self.app._create_component(
                                "sceneeditorpromptinput",
                                "textbox",
                                {"label": "Manual Text Prompt", "info": "Type what to track if auto-detection fails."},
                            )
                            self.app._create_component(
                                "scenerecomputebutton", "button", {"value": "▶️ Recompute with Prompt"}
                            )
                            self.app._create_component(
                                "scene_editor_subject_id", "textbox", {"visible": False, "value": ""}
                            )

            # 2. Filters
            with gr.Accordion("🔍 Batch Filter Scenes", open=False):
                self.app._create_component(
                    "scene_filter_status",
                    "markdown",
                    {"value": "*Apply constraints to automatically exclude bad scenes.*"},
                )
                with gr.Row():
                    self.app._create_component(
                        "scene_mask_area_min_input",
                        "slider",
                        {
                            "label": "Min Subject Size (%)",
                            "minimum": 0.0,
                            "maximum": 100.0,
                            "value": self.config.default_min_mask_area_pct,
                            "step": 0.1,
                            "info": "Remove scenes where subject is too small.",
                        },
                    )
                    self.app._create_component(
                        "scene_face_sim_min_input",
                        "slider",
                        {
                            "label": "Min Face Match",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "value": 0.0,
                            "step": 0.05,
                            "info": "Strictness of identity match.",
                        },
                    )
                    self.app._create_component(
                        "scene_quality_score_min_input",
                        "slider",
                        {
                            "label": "Min Quality Score",
                            "minimum": 0.0,
                            "maximum": 20.0,
                            "value": 0.0,
                            "step": 0.5,
                            "info": "Remove blurry/bad composition.",
                        },
                    )

            # 3. Main Gallery
            with gr.Group():
                # Pagination Controls
                with gr.Row(elem_id="pagination_row", equal_height=True):
                    with gr.Column(scale=2):
                        self.app._create_component(
                            "scene_gallery_view_toggle",
                            "radio",
                            {
                                "label": "View Mode",
                                "choices": self.app.SCENE_GALLERY_VIEW_CHOICES or ["Kept", "Rejected", "All"],
                                "value": "Kept",
                                "container": False,
                                "show_label": True,
                            },
                        )
                    with gr.Column(scale=3):
                        with gr.Row():
                            self.app._create_component(
                                "prev_page_button", "button", {"value": "⬅️ Previous", "size": "sm"}
                            )
                            self.app._create_component(
                                "page_number_input",
                                "dropdown",
                                {
                                    "label": "Page",
                                    "value": "1",
                                    "choices": ["1"],
                                    "interactive": True,
                                    "container": False,
                                    "scale": 0,
                                    "min_width": 80,
                                },
                            )
                            self.app._create_component("total_pages_label", "markdown", {"value": "/ 1 pages"})
                            self.app._create_component("next_page_button", "button", {"value": "Next ➡️", "size": "sm"})
                    with gr.Column(scale=1):
                        self.app._create_component("sceneundobutton", "button", {"value": "↩️ Undo", "size": "sm"})

                self.app.components["scene_gallery"] = gr.Gallery(
                    label="Scene Overview",
                    columns=8,
                    rows=2,
                    height=600,
                    show_label=False,
                    allow_preview=False,
                    container=True,
                    object_fit="contain",
                )

                with gr.Accordion("Display Settings", open=False):
                    with gr.Row():
                        self.app._create_component(
                            "scene_gallery_columns",
                            "slider",
                            {"label": "Columns", "minimum": 2, "maximum": 12, "value": 8, "step": 1},
                        )
                        self.app._create_component(
                            "scene_gallery_height",
                            "slider",
                            {"label": "Gallery Height (px)", "minimum": 200, "maximum": 1000, "value": 600, "step": 40},
                        )

            # 4. Action
            gr.Markdown("### 🚀 Ready?")
            self.app._create_component(
                "propagate_masks_button",
                "button",
                {"value": "⚡ Propagate Masks to All Frames", "variant": "primary", "interactive": False, "size": "lg", "visible": False},
            )
