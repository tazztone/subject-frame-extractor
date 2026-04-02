from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr

from core.application_state import ApplicationState
from core.events import ExtractionEvent, PropagationEvent, SessionLoadEvent
from core.models import Scene
from core.pipelines import (
    execute_analysis,
    execute_extraction,
    execute_pre_analysis,
    execute_propagation,
    execute_session_load,
)
from core.scene_utils import get_scene_status_text
from ui.decorators import safe_ui_callback
from ui.gallery_utils import build_scene_gallery_items

if TYPE_CHECKING:
    from ui.app_ui import AppUI


class PipelineHandler:
    """
    Handles pipeline execution triggers and callbacks for the UI.
    Extracted from monolith app_ui.py to improve maintainability.
    """

    def __init__(self, app: "AppUI"):
        self.app = app
        self.config = app.config
        self.logger = app.logger
        self.thumbnail_manager = app.thumbnail_manager
        self.model_registry = app.model_registry

    @safe_ui_callback("Extraction")
    def run_extraction_wrapper(self, current_state: ApplicationState, *args, progress=None):
        """Wrapper to execute the frame extraction pipeline."""
        event_dict = dict(zip(self.app.ext_ui_map_keys, args))
        event = ExtractionEvent(**event_dict)

        # Handle video source path logic if uploaded
        if "source_path" in event_dict and "upload_video" in event_dict:
            src = event_dict["source_path"]
            up = event_dict["upload_video"]
            if up and not src:
                event.source_path = up
            elif up and src and up != src:
                # If both are present, prefer the manual path if it exists, else upload
                if not Path(src).exists():
                    event.source_path = up

        for update in self.app._run_pipeline(
            execute_extraction,
            event,
            progress or gr.Progress(),
            lambda res: self._on_extraction_success(res, current_state),
        ):
            if isinstance(update, dict) and self.app.components["application_state"] in update:
                from typing import cast

                current_state = cast(ApplicationState, update[self.app.components["application_state"]])
            yield update

    def _on_extraction_success(self, result: dict, current_state: ApplicationState) -> dict:
        """Callback for successful extraction."""
        new_state = current_state.model_copy()
        new_state.extracted_video_path = result.get("extracted_video_path_state", "")
        new_state.extracted_frames_dir = result["extracted_frames_dir_state"]

        msg = f"""<div class="success-card">
        <h3>Extraction Complete</h3>
        <p>Frames extracted to: <code>{result["extracted_frames_dir_state"]}</code></p>
        <p><strong>Moving to next step automatically...</strong></p>
        </div>"""

        return {
            self.app.components["application_state"]: new_state,
            self.app.components["unified_status"]: msg,
            self.app.components["unified_log"]: result.get("unified_log", "Extraction Complete."),
            self.app.components["main_tabs"]: gr.update(selected=1),
        }

    @safe_ui_callback("Pre-Analysis")
    def run_pre_analysis_wrapper(self, current_state: ApplicationState, *args, progress=None):
        """Wrapper to execute the pre-analysis pipeline."""
        event = self.app._create_pre_analysis_event(current_state, *args)
        for update in self.app._run_pipeline(
            execute_pre_analysis,
            event,
            progress or gr.Progress(),
            lambda res: self._on_pre_analysis_success(res, current_state),
        ):
            if isinstance(update, dict) and self.app.components["application_state"] in update:
                from typing import cast

                current_state = cast(ApplicationState, update[self.app.components["application_state"]])
            yield update

    def _on_pre_analysis_success(self, result: dict, current_state: ApplicationState) -> dict:
        """Callback for successful pre-analysis."""
        new_state = current_state.model_copy()
        new_state.scenes = result["scenes"]
        new_state.analysis_output_dir = result["output_dir"]
        # Ensure video path is preserved
        if "video_path" in result:
            new_state.extracted_video_path = result["video_path"]

        # Auto-save logs
        self.app._save_session_log(result["output_dir"])

        scenes_objs = [Scene(**s) for s in result["scenes"]]
        status_text, button_update = get_scene_status_text(scenes_objs)

        # Hide propagation for image-only folders
        if not new_state.extracted_video_path:
            button_update = gr.update(visible=False)
        else:
            if isinstance(button_update, dict):
                button_update["visible"] = True
            else:
                button_update = gr.update(visible=True, interactive=True)

        if not new_state.extracted_video_path:
            msg = f"""<div class="success-card">
            <h3>Pre-Analysis Complete</h3>
            <p>Found <strong>{len(scenes_objs)}</strong> scenes in image folder.</p>
            <p><strong>Moving to select scenes automatically...</strong></p>
            </div>"""
        else:
            msg = f"""<div class="success-card">
            <h3>Pre-Analysis Complete</h3>
            <p>Found <strong>{len(scenes_objs)}</strong> scenes.</p>
            <p><strong>Moving to select fields automatically...</strong></p>
            </div>"""

        items, index_map, total_pages = build_scene_gallery_items(
            new_state.scenes, "Kept", new_state.extracted_frames_dir, page_num=1, config=self.config
        )
        new_state.scene_gallery_index_map = index_map
        page_choices = [str(i) for i in range(1, total_pages + 1)] if total_pages > 0 else ["1"]

        # Merged updates
        updates = {
            self.app.components["application_state"]: new_state,
            self.app.components["scene_filter_status"]: status_text,
            self.app.components["unified_status"]: msg,
            self.app.components["scene_gallery"]: gr.update(value=items),
            self.app.components["total_pages_label"]: f"/ {total_pages} pages",
            self.app.components["page_number_input"]: gr.update(choices=page_choices, value="1"),
            self.app.components["unified_log"]: result.get("unified_log", "Pre-Analysis Complete."),
        }

        # Include any explicit UI updates from the pipeline result (e.g. visibility toggles)
        for k, v in result.items():
            if k in self.app.components:
                updates[self.app.components[k]] = v

        # These specific components must be visible for the next step
        updates[self.app.components["seeding_results_column"]] = gr.update(visible=True)
        updates[self.app.components["propagation_group"]] = gr.update(visible=True)
        updates[self.app.components["propagate_masks_button"]] = button_update
        updates[self.app.components["main_tabs"]] = gr.update(selected=2)

        return updates

    def _propagation_button_handler(self, current_state: ApplicationState):
        """Unified guard for propagation button."""
        if not current_state.extracted_video_path:
            yield {
                self.app.components[
                    "unified_log"
                ]: "Note: Propagation is not needed for image folders. Proceed to Compute Metrics."
            }
            return

        # If valid, just return the state to trigger the event chain
        yield {self.app.components["application_state"]: current_state}

    @safe_ui_callback("Propagation")
    def run_propagation_wrapper(self, current_state: ApplicationState, *args, progress=None):
        """Wrapper to execute the mask propagation pipeline."""
        scenes = current_state.scenes
        if not scenes:
            yield {self.app.components["unified_log"]: "No scenes."}
            return
        params = self.app._create_pre_analysis_event(current_state, *args)
        event = PropagationEvent(
            output_folder=params.output_folder, video_path=params.video_path, scenes=scenes, analysis_params=params
        )
        yield from self.app._run_pipeline(
            execute_propagation,
            event,
            progress or gr.Progress(),
            lambda res: self._on_propagation_success(res, current_state),
        )

    def _on_propagation_success(self, result: dict, current_state: ApplicationState) -> dict:
        """Callback for successful propagation."""
        msg = """<div class="success-card"><h3>Mask Propagation Complete</h3></div>"""
        return {
            self.app.components["application_state"]: current_state,
            self.app.components["unified_status"]: msg,
            self.app.components["unified_log"]: result.get("unified_log", "Propagation Complete."),
        }

    @safe_ui_callback("Analysis")
    def run_analysis_wrapper(self, current_state: ApplicationState, *args, progress=None):
        """Wrapper to execute the full analysis pipeline."""
        scenes = current_state.scenes
        if not scenes:
            yield {self.app.components["unified_log"]: "No scenes."}
            return
        params = self.app._create_pre_analysis_event(current_state, *args)
        event = PropagationEvent(
            output_folder=params.output_folder, video_path=params.video_path, scenes=scenes, analysis_params=params
        )
        yield from self.app._run_pipeline(
            execute_analysis,
            event,
            progress or gr.Progress(),
            lambda res: self._on_analysis_success(res, current_state),
        )

    def _on_analysis_success(self, result: dict, current_state: ApplicationState) -> dict:
        """Callback for successful analysis."""
        new_state = current_state.model_copy()
        new_state.analysis_metadata_path = result["metadata_path"]
        self.app._save_session_log(str(Path(result["metadata_path"]).parent))

        msg = """<div class="success-card"><h3>Analysis Complete</h3></div>"""
        return {
            self.app.components["application_state"]: new_state,
            self.app.components["unified_status"]: msg,
            self.app.components["unified_log"]: result.get("unified_log", "Analysis Complete."),
        }

    @safe_ui_callback("Load Session")
    def run_session_load_wrapper(self, session_path: str, current_state: ApplicationState):
        """Loads a previous session and updates the UI state."""
        event = SessionLoadEvent(session_path=session_path)
        yield {self.app.components["unified_status"]: "Loading Session..."}

        result = execute_session_load(event, self.logger)
        if result.get("error"):
            yield {self.app.components["unified_log"]: f"[ERROR] {result['error']}"}
            return

        run_config = result["run_config"]
        session_path_obj = Path(result["session_path"])
        scenes_data = result["scenes"]
        metadata_exists = result["metadata_exists"]

        def _resolve_output_dir(base: Path, output_folder: str | None) -> Path | None:
            if not output_folder:
                return None
            p = Path(output_folder)
            if p.exists():
                return p.resolve()
            if not p.is_absolute():
                return (base / p).resolve()
            return p

        output_dir = _resolve_output_dir(session_path_obj, run_config.get("output_folder")) or session_path_obj

        new_state = current_state.model_copy()
        new_state.extracted_video_path = run_config.get("video_path", "")
        new_state.extracted_frames_dir = str(output_dir)
        new_state.analysis_output_dir = str(output_dir.resolve() if output_dir else "")

        updates = {
            self.app.components["source_input"]: gr.update(value=run_config.get("source_path", "")),
            self.app.components["max_resolution"]: gr.update(value=run_config.get("max_resolution", "1080")),
            self.app.components["thumb_megapixels_input"]: gr.update(value=run_config.get("thumb_megapixels", 0.5)),
            self.app.components["ext_scene_detect_input"]: gr.update(value=run_config.get("scene_detect", True)),
            self.app.components["method_input"]: gr.update(value=run_config.get("method", "scene")),
            self.app.components["pre_analysis_enabled_input"]: gr.update(
                value=run_config.get("pre_analysis_enabled", True)
            ),
            self.app.components["pre_sample_nth_input"]: gr.update(value=run_config.get("pre_sample_nth", 1)),
            self.app.components["compute_face_sim"]: gr.update(
                value=run_config.get("compute_face_sim", run_config.get("enable_face_filter", False))
            ),
            self.app.components["face_model_name_input"]: gr.update(
                value=run_config.get("face_model_name", "buffalo_l")
            ),
            self.app.components["face_ref_img_path_input"]: gr.update(value=run_config.get("face_ref_img_path", "")),
            self.app.components["text_prompt_input"]: gr.update(value=run_config.get("text_prompt", "")),
            self.app.components["best_frame_strategy_input"]: gr.update(
                value=run_config.get("best_frame_strategy", "Largest Person")
            ),
            self.app.components["tracker_model_name_input"]: gr.update(
                value=run_config.get("tracker_model_name", self.config.default_tracker_model_name)
            ),
            self.app.components["application_state"]: new_state,
        }

        if "seed_strategy" in run_config:
            updates[self.app.components["best_frame_strategy_input"]] = gr.update(value=run_config["seed_strategy"])
        if "primary_seed_strategy" in run_config:
            val = run_config["primary_seed_strategy"]
            if isinstance(val, str):
                import re

                val = re.sub(r"^[^\w\s]+\s+", "", val)
            updates[self.app.components["primary_seed_strategy_input"]] = gr.update(value=val)

        if scenes_data and output_dir:
            from core.scene_utils import get_scene_status_text

            scenes = [Scene(**s) for s in scenes_data]
            status_text, button_update = get_scene_status_text(scenes)
            gallery_items, index_map, _ = build_scene_gallery_items(scenes, "Kept", str(output_dir), config=self.config)
            new_state.scenes = [s.model_dump() for s in scenes]
            new_state.scene_gallery_index_map = index_map

            updates.update(
                {
                    self.app.components["propagate_masks_button"]: button_update,
                    self.app.components["seeding_results_column"]: gr.update(visible=True),
                    self.app.components["propagation_group"]: gr.update(visible=True),
                    self.app.components["scene_filter_status"]: status_text,
                    self.app.components["scene_face_sim_min_input"]: gr.update(
                        visible=any((s.seed_metrics or {}).get("best_face_sim") is not None for s in scenes)
                    ),
                    self.app.components["scene_gallery"]: gr.update(value=gallery_items),
                }
            )

        if metadata_exists:
            new_state.analysis_output_dir = str(session_path_obj)
            new_state.analysis_metadata_path = str(session_path_obj / "metadata.db")
            updates.update(
                {
                    self.app.components["filtering_tab"]: gr.update(interactive=True),
                }
            )

        # Update compute metrics checkboxes
        for metric in self.app.ana_ui_map_keys:
            if metric.startswith("compute_") and metric in self.app.components:
                updates[self.app.components[metric]] = gr.update(value=run_config.get(metric, True))

        updates.update(
            {
                self.app.components["application_state"]: new_state,
                self.app.components["unified_log"]: f"Successfully loaded session from: {session_path}",
                self.app.components["unified_status"]: "Session Loaded.",
            }
        )
        yield updates
