from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

from core.enums import ANCHOR_STRATEGIES, COCO_CLASSES

if TYPE_CHECKING:
    from ui.app_ui import AppUI


class SubjectTabBuilder:
    def __init__(self, app: "AppUI"):
        self.app = app
        self.config = app.config

    def build(self):
        """Creates the content for the 'Subject' tab."""
        self.app._create_section_header("Step 2: Define Subject", "Specify the target for tracking and extraction.")

        # 1. Strategy Selection (Always Visible)
        with gr.Group():
            gr.Markdown("#### 1. Tracking Strategy")
            self.app._reg(
                "primary_seed_strategy",
                self.app._create_component(
                    "primary_seed_strategy_input",
                    "radio",
                    {
                        "choices": self.app.PRIMARY_SEED_STRATEGY_CHOICES,
                        "value": self.config.default_primary_seed_strategy,
                        "label": "How to find the subject?",
                        "info": "Choose 'Automatic' for general subjects, 'By Face' for specific identity.",
                        "show_label": False,
                        "elem_id": "primary_seed_strategy_input",
                    },
                ),
            )

        # 2. Dynamic Input Groups (Toggled by Radio)

        # --- A. Face Seeding Group ---
        with gr.Group(visible=False) as face_seeding_group:
            self.app.components["face_seeding_group"] = face_seeding_group
            gr.Markdown("#### 2. Provide Face Reference")

            with gr.Tabs():
                # Tab 1: Upload
                with gr.Tab("Upload Photo"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            self.app._reg(
                                "face_ref_img_upload",
                                self.app._create_component(
                                    "face_ref_img_upload_input",
                                    "file",
                                    {
                                        "label": "Upload Reference Photo",
                                        "type": "filepath",
                                        "height": 100,
                                        "file_types": ["image"],
                                        "elem_id": "face_ref_img_upload_input",
                                    },
                                ),
                            )
                        with gr.Column(scale=1):
                            self.app._create_component(
                                "face_ref_image",
                                "image",
                                {
                                    "label": "Reference Face Preview",
                                    "interactive": False,
                                    "height": 150,
                                    "show_label": False,
                                },
                            )

                # Tab 2: Scan Video (Discovery)
                with gr.Tab("Scan Video for Subjects", elem_id="scan_video_tab"):
                    gr.Markdown(
                        "### 1. Scan Video\nClick **'Scan Video Now'**. The AI will find valid subjects in the footage."
                    )

                    with gr.Row():
                        self.app._create_component(
                            "find_people_button",
                            "button",
                            {
                                "value": "🔍 Scan Video Now",
                                "variant": "secondary",
                                "scale": 1,
                                "elem_id": "find_people_button",
                            },
                        )
                        self.app._create_component(
                            "identity_confidence_slider",
                            "slider",
                            {
                                "label": "Clustering Strictness",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "step": 0.05,
                                "value": 0.5,
                                "info": "Higher = Stricter grouping.",
                                "scale": 3,
                            },
                        )

                    self.app._create_component("find_people_status", "markdown", {"value": ""})

                    with gr.Group(visible=False) as discovered_people_group:
                        self.app.components["discovered_people_group"] = discovered_people_group
                        gr.Markdown(
                            "### 2. Select Subject\n**Click on an object** below to select your target. A confirmation will appear."
                        )
                        self.app._create_component(
                            "discovered_faces_gallery",
                            "gallery",
                            {
                                "label": "Detected Subjects (Click to Select)",
                                "columns": [2, 4, 6],
                                "height": "auto",
                                "allow_preview": False,
                                "object_fit": "cover",
                            },
                        )

            with gr.Accordion("Advanced: Use Local File Path", open=False):
                self.app._reg(
                    "face_ref_img_path",
                    self.app._create_component(
                        "face_ref_img_path_input",
                        "textbox",
                        {
                            "label": "Absolute Path to Reference Image",
                            "placeholder": "/path/to/image.jpg",
                            "visible": True,
                            "info": "Useful for automation or server-side files.",
                        },
                    ),
                )

        # --- B. Text Seeding Group ---
        with gr.Group(visible=False) as text_seeding_group:
            self.app.components["text_seeding_group"] = text_seeding_group
            gr.Markdown("#### 2. Text Description")
            self.app._reg(
                "text_prompt",
                self.app._create_component(
                    "text_prompt_input",
                    "textbox",
                    {
                        "label": "What should we look for?",
                        "placeholder": "e.g., 'a man in a blue suit', 'a red car'",
                        "info": "Be specific. Color and clothing help. Example: 'person in red shirt'",
                        "lines": 1,
                        "show_label": False,
                    },
                ),
            )

        # --- C. Auto/General Seeding Group ---
        with gr.Group(visible=True) as auto_seeding_group:
            self.app.components["auto_seeding_group"] = auto_seeding_group
            # Only show this if strictly necessary for the strategy, otherwise it adds noise.
            # Currently reused for 'Find Prominent Subject'.
            with gr.Row():
                self.app._reg(
                    "best_frame_strategy",
                    self.app._create_component(
                        "best_frame_strategy_input",
                        "dropdown",
                        {
                            "choices": self.app.SEED_STRATEGY_CHOICES,
                            "value": self.config.default_seed_strategy,
                            "label": "Anchor Selection Strategy",
                            "info": "If multiple frames are candidates, which one is used as the tracking anchor?",
                            "elem_id": "best_frame_strategy_input",
                        },
                    ),
                )

        # 3. Hidden State Components (Required for Logic)
        self.app._create_component("person_radio", "radio", {"label": "Select Person", "choices": [], "visible": False})
        self.app._reg(
            "resume",
            self.app._create_component(
                "resume_input", "checkbox", {"label": "Resume", "value": self.config.default_resume, "visible": False}
            ),
        )
        self.app._reg(
            "enable_subject_mask",
            self.app._create_component(
                "enable_subject_mask_input",
                "checkbox",
                {"label": "Enable Subject Mask", "value": self.config.default_enable_subject_mask, "visible": False},
            ),
        )
        self.app._reg(
            "min_mask_area_pct",
            self.app._create_component(
                "min_mask_area_pct_input",
                "slider",
                {"label": "Min Mask Area Pct", "value": self.config.default_min_mask_area_pct, "visible": False},
            ),
        )
        self.app._reg(
            "sharpness_base_scale",
            self.app._create_component(
                "sharpness_base_scale_input",
                "slider",
                {
                    "label": "Sharpness Base Scale",
                    "minimum": 0,
                    "maximum": 5000,
                    "step": 100,
                    "value": self.config.default_sharpness_base_scale,
                    "visible": False,
                },
            ),
        )
        self.app._reg(
            "edge_strength_base_scale",
            self.app._create_component(
                "edge_strength_base_scale_input",
                "slider",
                {
                    "label": "Edge Strength Base Scale",
                    "minimum": 0,
                    "maximum": 1000,
                    "step": 10,
                    "value": self.config.default_edge_strength_base_scale,
                    "visible": False,
                },
            ),
        )

        # 4. Action Button
        self.app._create_component(
            "start_pre_analysis_button",
            "button",
            {
                "value": "Confirm Subject & Find Scenes (Next Step)",
                "variant": "primary",
                "size": "lg",
                "elem_id": "start_pre_analysis_button",
            },
        )

        # 5. Advanced Configuration Accordion
        with gr.Accordion("Advanced Model Configuration", open=False, elem_id="subject_advanced_config_accordion"):
            with gr.Row():
                self.app._reg(
                    "tracker_model_name",
                    self.app._create_component(
                        "tracker_model_name_input",
                        "dropdown",
                        {
                            "choices": self.app.TRACKER_MODEL_CHOICES,
                            "value": self.config.default_tracker_model_name,
                            "label": "Tracking Model",
                            "info": "SAM3 is slower but state-of-the-art.",
                        },
                    ),
                )
                self.app._reg(
                    "face_model_name",
                    self.app._create_component(
                        "face_model_name_input",
                        "dropdown",
                        {
                            "choices": self.app.FACE_MODEL_NAME_CHOICES,
                            "value": self.config.default_face_model_name,
                            "label": "Face Model",
                            "info": "Buffalo_L (Accurate) vs Buffalo_S (Fast).",
                        },
                    ),
                )

            with gr.Row():
                self.app._reg(
                    "subject_detector_model",
                    self.app._create_component(
                        "subject_detector_model_input",
                        "dropdown",
                        {
                            "choices": self.app.SUBJECT_DETECTOR_MODEL_CHOICES,
                            "value": self.config.default_subject_detector_model,
                            "label": "Subject Detector Fallback",
                            "info": "Used by SAM2 + Automatic mode.",
                        },
                    ),
                )
                self.app._reg(
                    "subject_detector_class_name",
                    self.app._create_component(
                        "subject_detector_class_input",
                        "dropdown",
                        {
                            "choices": COCO_CLASSES,
                            "value": self.config.default_subject_detector_class,
                            "label": "Subject Class",
                            "info": "Target object type for YOLO detection.",
                            "visible": self.config.default_subject_detector_model == "YOLO26n",
                            "filterable": True,
                        },
                    ),
                )
                self.app._reg(
                    "subject_detector_threshold",
                    self.app._create_component(
                        "subject_detector_threshold_input",
                        "slider",
                        {
                            "label": "Detector Threshold",
                            "minimum": 0.1,
                            "maximum": 0.9,
                            "step": 0.05,
                            "value": self.config.default_subject_detector_threshold,
                            "info": "Lower = More sensitive.",
                        },
                    ),
                )

            with gr.Row():
                self.app._reg(
                    "pre_analysis_enabled",
                    self.app._create_component(
                        "pre_analysis_enabled_input",
                        "checkbox",
                        {
                            "label": "Enable Pre-Analysis Scan",
                            "value": self.config.default_pre_analysis_enabled,
                            "info": "Scans thumbnails first to find best shots.",
                        },
                    ),
                )
                self.app._reg(
                    "pre_sample_nth",
                    self.app._create_component(
                        "pre_sample_nth_input",
                        "number",
                        {
                            "label": "Scan Step Rate",
                            "value": self.config.default_pre_sample_nth,
                            "info": "Process every Nth thumbnail during scan.",
                        },
                    ),
                )

        # 6. Propagation Placeholder (Filled later)
        with gr.Group(visible=False) as propagation_group:
            self.app.components["propagation_group"] = propagation_group

        # 7. Event Handlers for Dynamic Subject Logic
        c = self.app.components

        def handle_detector_change(detector):
            return gr.update(visible=(detector and detector.startswith("YOLO26")))

        c["subject_detector_model_input"].change(
            handle_detector_change,
            inputs=[c["subject_detector_model_input"]],
            outputs=[c["subject_detector_class_input"]],
        )

        def handle_class_change(class_name, current_strategy):
            # Only person class supports face-based seeding
            choices = ANCHOR_STRATEGIES
            if class_name != "person":
                choices = [s for s in ANCHOR_STRATEGIES if s != "Best Face"]

            new_val = current_strategy
            if class_name != "person" and current_strategy == "Best Face":
                new_val = "Largest Subject"

            return gr.update(choices=choices, value=new_val)

        c["subject_detector_class_input"].change(
            handle_class_change,
            inputs=[c["subject_detector_class_input"], c["best_frame_strategy_input"]],
            outputs=[c["best_frame_strategy_input"]],
        )
