from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

if TYPE_CHECKING:
    from ui.app_ui import AppUI


class MetricsTabBuilder:
    def __init__(self, app: "AppUI"):
        self.app = app
        self.config = app.config

    def build(self):
        """Creates the content for the 'Metrics' tab."""
        self.app._create_section_header(
            "Step 4: Analysis Metrics", "Choose what properties to calculate for each frame."
        )

        with gr.Row():
            with gr.Column():
                with gr.Group():
                    gr.Markdown("#### ✨ Visual Quality")
                    self.app._reg(
                        "compute_quality_score",
                        self.app._create_component(
                            "compute_quality_score",
                            "checkbox",
                            {
                                "label": "Quality Score (Recommended)",
                                "value": True,
                                "info": "Composite score of sharpness, face visibility, and composition.",
                            },
                        ),
                    )
                    self.app._reg(
                        "compute_sharpness",
                        self.app._create_component(
                            "compute_sharpness",
                            "checkbox",
                            {"label": "Sharpness", "value": True, "info": "Detects blurriness and fine detail."},
                        ),
                    )
                    try:
                        import pyiqa

                        niqe_avail = pyiqa is not None
                    except ImportError:
                        niqe_avail = False

                    self.app._reg(
                        "compute_niqe",
                        self.app._create_component(
                            "compute_niqe",
                            "checkbox",
                            {
                                "label": "NIQE (Natural Image Quality)",
                                "value": False,
                                "interactive": niqe_avail,
                                "info": "No-reference image quality assessment. (Slow but accurate).",
                            },
                        ),
                    )

            with gr.Column():
                with gr.Group():
                    gr.Markdown("#### 👤 Subject & Content")
                    self.app._reg(
                        "compute_face_sim",
                        self.app._create_component(
                            "compute_face_sim",
                            "checkbox",
                            {
                                "label": "Face Identity Match",
                                "value": True,
                                "info": "How much does the face look like the reference?",
                            },
                        ),
                    )
                    self.app._reg(
                        "compute_eyes_open",
                        self.app._create_component(
                            "compute_eyes_open",
                            "checkbox",
                            {
                                "label": "Eyes Open Score",
                                "value": True,
                                "info": "Detects blinking (1.0 = Open, 0.0 = Closed).",
                            },
                        ),
                    )
                    self.app._reg(
                        "compute_subject_mask_area",
                        self.app._create_component(
                            "compute_subject_mask_area",
                            "checkbox",
                            {
                                "label": "Subject Size (%)",
                                "value": True,
                                "info": "Percentage of the frame occupied by the subject.",
                            },
                        ),
                    )

            with gr.Column():
                with gr.Group():
                    gr.Markdown("#### 📐 Geometry & Composition")
                    self.app._reg(
                        "compute_yaw",
                        self.app._create_component(
                            "compute_yaw",
                            "checkbox",
                            {
                                "label": "Head Yaw (L/R)",
                                "value": False,
                                "info": "Head rotation: Left vs Right profile.",
                            },
                        ),
                    )
                    self.app._reg(
                        "compute_pitch",
                        self.app._create_component(
                            "compute_pitch",
                            "checkbox",
                            {
                                "label": "Head Pitch (Up/Down)",
                                "value": False,
                                "info": "Head rotation: Looking up vs down.",
                            },
                        ),
                    )

        with gr.Accordion("🔧 Advanced / Legacy Metrics", open=False):
            with gr.Row():
                with gr.Column():
                    self.app._reg(
                        "compute_edge_strength",
                        self.app._create_component(
                            "compute_edge_strength",
                            "checkbox",
                            {"label": "Edge Strength", "value": False, "info": "Overall contrast of edges."},
                        ),
                    )
                    self.app._reg(
                        "compute_contrast",
                        self.app._create_component(
                            "compute_contrast",
                            "checkbox",
                            {"label": "Contrast", "value": False, "info": "Luma variance."},
                        ),
                    )
                with gr.Column():
                    self.app._reg(
                        "compute_brightness",
                        self.app._create_component(
                            "compute_brightness",
                            "checkbox",
                            {"label": "Brightness", "value": False, "info": "Average pixel intensity."},
                        ),
                    )
                    self.app._reg(
                        "compute_entropy",
                        self.app._create_component(
                            "compute_entropy",
                            "checkbox",
                            {"label": "Entropy", "value": False, "info": "Information density."},
                        ),
                    )

        with gr.Accordion("📂 Deduplication Preparation", open=False):
            self.app._reg(
                "compute_phash",
                self.app._create_component(
                    "compute_phash",
                    "checkbox",
                    {
                        "label": "Compute Perceptual Hash (p-Hash)",
                        "value": True,
                        "info": "Required for identifying duplicate frames later.",
                    },
                ),
            )

        self.app.components["start_analysis_button"] = gr.Button("⚡ Run Analysis", variant="primary", size="lg")
