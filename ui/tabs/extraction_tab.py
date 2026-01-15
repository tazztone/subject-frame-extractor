from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

if TYPE_CHECKING:
    from ui.app_ui import AppUI


class ExtractionTabBuilder:
    def __init__(self, app: "AppUI"):
        self.app = app
        self.config = app.config

    def build(self):
        """Creates the content for the 'Source' tab."""
        self.app._create_section_header(
            "Step 1: Input & Extraction", "Select your video source and how you want to process it."
        )

        # 1. Source Selection
        with gr.Group():
            gr.Markdown("#### 📂 Source Selection")
            with gr.Tabs():
                with gr.Tab("🔗 Path / URL"):
                    with gr.Row():
                        with gr.Column(scale=4):
                            self.app._reg(
                                "source_path",
                                self.app._create_component(
                                    "source_input",
                                    "textbox",
                                    {
                                        "label": "Input Path or URL",
                                        "placeholder": "Paste YouTube URL or local path (file/folder)...",
                                        "info": "Enter a path to a video file, a folder of images, or a YouTube link.",
                                        "show_label": False,
                                        "container": False,
                                        "scale": 4,
                                    },
                                ),
                            )
                        with gr.Column(scale=1, min_width=150):
                            self.app._reg(
                                "max_resolution",
                                self.app._create_component(
                                    "max_resolution",
                                    "dropdown",
                                    {
                                        "choices": self.app.MAX_RESOLUTION_CHOICES,
                                        "value": self.config.default_max_resolution,
                                        "label": "YouTube Res",
                                        "info": "Only for YouTube downloads.",
                                        "show_label": True,
                                        "container": True,
                                        "scale": 1,
                                    },
                                ),
                            )

                with gr.Tab("⬆️ Upload File"):
                    self.app._reg(
                        "upload_video",
                        self.app._create_component(
                            "upload_video_input",
                            "file",
                            {
                                "label": "Upload Video File",
                                "file_count": "multiple",
                                "file_types": ["video"],
                                "type": "filepath",
                                "height": 80,
                            },
                        ),
                    )

        # 2. Strategy & Settings
        with gr.Group():
            gr.Markdown("#### ⚙️ Extraction Strategy")
            with gr.Row(equal_height=True):
                with gr.Column(scale=2):
                    self.app._reg(
                        "method",
                        self.app._create_component(
                            "method_input",
                            "dropdown",
                            {
                                "choices": self.app.METHOD_CHOICES,
                                "value": self.config.default_method,
                                "label": "Extraction Method",
                                "info": "How frames are selected from the video.",
                            },
                        ),
                    )
                with gr.Column(scale=1):
                    self.app._reg(
                        "scene_detect",
                        self.app._create_component(
                            "ext_scene_detect_input",
                            "checkbox",
                            {
                                "label": "Split by Scenes",
                                "value": self.config.default_scene_detect,
                                "info": "Detect shot changes (cuts) automatically.",
                            },
                        ),
                    )

            # Dynamic settings based on method
            with gr.Row():
                self.app._reg(
                    "interval",
                    self.app._create_component(
                        "interval_input",
                        "number",
                        {
                            "label": "Interval (seconds)",
                            "value": self.config.default_interval,
                            "minimum": 0.1,
                            "step": 0.1,
                            "visible": self.config.default_method == "interval",
                            "info": "Extract one frame every X seconds.",
                        },
                    ),
                )
                self.app._reg(
                    "nth_frame",
                    self.app._create_component(
                        "nth_frame_input",
                        "number",
                        {
                            "label": "N-th Frame",
                            "value": self.config.default_nth_frame,
                            "minimum": 1,
                            "step": 1,
                            "visible": self.config.default_method in ["every_nth_frame", "nth_plus_keyframes"],
                            "info": "Extract every Nth frame (e.g., 10 = 10% of video).",
                        },
                    ),
                )

        # 3. Advanced Settings (Hidden by default)
        with gr.Accordion("🔧 Advanced Processing Settings", open=False):
            with gr.Group(visible=True) as thumbnail_group:
                self.app.components["thumbnail_group"] = thumbnail_group
                self.app._reg(
                    "thumb_megapixels",
                    self.app._create_component(
                        "thumb_megapixels_input",
                        "slider",
                        {
                            "label": "Analysis Resolution (Megapixels)",
                            "minimum": 0.1,
                            "maximum": 2.0,
                            "step": 0.1,
                            "value": self.config.default_thumb_megapixels,
                            "info": "Lower = Faster, Higher = Better small object detection. Default 0.5 is usually good.",
                        },
                    ),
                )

        # 4. Action Area
        with gr.Row(elem_id="extraction_actions"):
            self.app.components["start_extraction_button"] = gr.Button(
                "🚀 Start Extraction", variant="primary", scale=2, size="lg"
            )
            self.app._create_component(
                "add_to_queue_button",
                "button",
                {"value": "➕ Queue for Batch", "variant": "secondary", "scale": 1, "size": "lg"},
            )

        # 5. Batch Queue
        with gr.Accordion("📚 Batch Queue", open=False) as batch_accordion:
            self.app.components["batch_accordion"] = batch_accordion
            gr.Markdown("*Process multiple videos in the background.*")
            self.app._create_component(
                "batch_queue_dataframe",
                "dataframe",
                {
                    "headers": ["Path", "Status", "Progress", "Message"],
                    "datatype": ["str", "str", "number", "str"],
                    "interactive": False,
                    "value": [],
                },
            )
            with gr.Row():
                self.app._create_component(
                    "start_batch_button", "button", {"value": "▶️ Run Queue", "variant": "primary"}
                )
                self.app._create_component("stop_batch_button", "button", {"value": "⏹️ Stop Queue", "variant": "stop"})
                self.app._create_component("clear_queue_button", "button", {"value": "🗑️ Clear Queue"})
            self.app._create_component(
                "batch_workers_slider",
                "slider",
                {
                    "label": "Max Parallel Jobs",
                    "minimum": 1,
                    "maximum": 4,
                    "value": 1,
                    "step": 1,
                    "info": "Be careful with VRAM usage!",
                },
            )
