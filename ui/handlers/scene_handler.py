from __future__ import annotations

import copy
from typing import TYPE_CHECKING, List, Dict, Deque, Optional, Tuple
from pathlib import Path
import gradio as gr

from core.models import Scene, SceneState
from core.scene_utils import (
    _create_analysis_context,
    _recompute_single_preview,
    _wire_recompute_handler,
    get_scene_status_text,
    save_scene_seeds,
    toggle_scene_status,
)
from ui.gallery_utils import build_scene_gallery_items
from core.application_state import ApplicationState

if TYPE_CHECKING:
    from ui.app_ui import AppUI


class SceneHandler:
    """
    Handles scene-related UI operations (selection, filtering, undo, etc.).
    """

    def __init__(self, app: "AppUI"):
        self.app = app
        self.config = app.config
        self.logger = app.logger
        self.thumbnail_manager = app.thumbnail_manager
        self.model_registry = app.model_registry

    def setup_handlers(self):
        """Configures event handlers for the scene selection tab (pagination, bulk actions)."""
        c = self.app.components

        def on_page_change(app_state: ApplicationState, view: str, page_num):
            """Handle page change - returns gallery items and dropdown update with choices."""
            try:
                current_page = int(page_num) if page_num else 1
            except (ValueError, TypeError):
                current_page = 1
            
            items, index_map, total_pages = build_scene_gallery_items(
                app_state.scenes, view, app_state.extracted_frames_dir, page_num=current_page, config=self.config
            )
            # Generate page choices for dropdown
            page_choices = [str(i) for i in range(1, total_pages + 1)] if total_pages > 0 else ["1"]
            
            app_state.scene_gallery_index_map = index_map
            
            return (
                app_state,
                gr.update(value=items),
                f"/ {total_pages} pages",
                gr.update(choices=page_choices, value=str(current_page)),
            )

        def on_view_change(app_state: ApplicationState, view: str):
            """Handle view filter change - reset to page 1 and update dropdown choices."""
            items, index_map, total_pages = build_scene_gallery_items(
                app_state.scenes, view, app_state.extracted_frames_dir, page_num=1, config=self.config
            )
            app_state.scene_gallery_index_map = index_map
            page_choices = [str(i) for i in range(1, total_pages + 1)] if total_pages > 0 else ["1"]
            return app_state, gr.update(value=items), f"/ {total_pages} pages", gr.update(choices=page_choices, value="1")

        def on_next_page(app_state: ApplicationState, view: str, page_num):
            """Go to next page (clamped to max)."""
            try:
                current = int(page_num) if page_num else 1
            except (ValueError, TypeError):
                current = 1
            _, _, total_pages = build_scene_gallery_items(
                app_state.scenes, view, app_state.extracted_frames_dir, page_num=1, config=self.config
            )
            new_page = min(current + 1, total_pages)
            return on_page_change(app_state, view, new_page)

        def on_prev_page(app_state: ApplicationState, view: str, page_num):
            """Go to previous page."""
            try:
                current = int(page_num) if page_num else 1
            except (ValueError, TypeError):
                current = 1
            return on_page_change(app_state, view, max(1, current - 1))

        # Recompute wrapper to handle app_state
        def on_recompute(app_state: ApplicationState, view, txt, *ana_args):
            # Update prompt from UI input just in case
            # Note: The prompt input is passed as `txt`.
            # We need to construct scene objects
            scenes_objs = [Scene(**s) for s in app_state.scenes]
            
            app_state.push_history(app_state.scenes) # Save history before recompute
            
            scenes_objs, gallery_items, index_map, status, _ = _wire_recompute_handler(
                self.config,
                self.logger,
                self.thumbnail_manager,
                scenes_objs,
                app_state.selected_scene_id,
                app_state.analysis_output_dir,
                txt,
                view,
                self.app.ana_ui_map_keys,
                list(ana_args),
                self.app.cuda_available,
                self.model_registry,
            )
            
            # Update state
            app_state.scenes = [s.model_dump() for s in scenes_objs]
            app_state.scene_gallery_index_map = index_map
            
            return (
                app_state,
                gr.update(value=gallery_items),
                status, # sceneeditorstatusmd
            )


        # --- Wire existing components ---
        
        # NOTE: We assume c["application_state"] exists.

        c["scene_gallery_view_toggle"].change(
            on_view_change,
            [c["application_state"], c["scene_gallery_view_toggle"]],
            [c["application_state"], c["scene_gallery"], c["total_pages_label"], c["page_number_input"]],
        )
        c["next_page_button"].click(
            on_next_page,
            [
                c["application_state"],
                c["scene_gallery_view_toggle"],
                c["page_number_input"],
            ],
            [c["application_state"], c["scene_gallery"], c["total_pages_label"], c["page_number_input"]],
        )
        c["prev_page_button"].click(
            on_prev_page,
            [
                c["application_state"],
                c["scene_gallery_view_toggle"],
                c["page_number_input"],
            ],
            [c["application_state"], c["scene_gallery"], c["total_pages_label"], c["page_number_input"]],
        )
        c["page_number_input"].change(
            on_page_change,
            [
                c["application_state"],
                c["scene_gallery_view_toggle"],
                c["page_number_input"],
            ],
            [c["application_state"], c["scene_gallery"], c["total_pages_label"], c["page_number_input"]],
        )

        c["scene_gallery"].select(
            self.on_select_for_edit,
            inputs=[
                c["application_state"],
                c["scene_gallery_view_toggle"],
            ],
            outputs=[
                c["application_state"],
                c["scene_filter_status"],
                c["scene_gallery"], # Updates just in case but usually not needed for select? Actually on_select_for_edit returns update() for gallery.
                c["sceneeditorstatusmd"],
                c["sceneeditorpromptinput"],
                c["scene_editor_group"],
                c["subject_selection_gallery"],
                c["propagate_masks_button"],
                c["gallery_image_preview"],
            ],
        )

        c["scenerecomputebutton"].click(
            on_recompute,
            inputs=[
                c["application_state"],
                c["scene_gallery_view_toggle"],
                c["sceneeditorpromptinput"],
                *self.app.ana_input_components,
            ],
            outputs=[
                c["application_state"],
                c["scene_gallery"],
                c["sceneeditorstatusmd"],
            ],
        )

        c["sceneresetbutton"].click(
            self.on_reset_scene_wrapper,
            inputs=[
                c["application_state"],
                c["scene_gallery_view_toggle"],
            ]
            + self.app.ana_input_components,
            outputs=[
                c["application_state"],
                c["scene_gallery"],
                c["sceneeditorstatusmd"],
            ],
        )

        c["sceneincludebutton"].click(
            lambda s, v: self.on_editor_toggle(s, v, "included"),
            # Lambda signature mismatch. on_editor_toggle(app_state, view, status)
            # Input list: [app_state, view_toggle, dummy_history?]
            # Wait, I removed history from on_editor_toggle.
            # So lambda should be: lambda s, v: self.on_editor_toggle(s, v, "included")
            inputs=[
                c["application_state"],
                c["scene_gallery_view_toggle"],
            ],
            outputs=[
                c["application_state"],
                c["scene_filter_status"],
                c["scene_gallery"],
                c["propagate_masks_button"],
            ],
        )
        c["sceneexcludebutton"].click(
            lambda s, v: self.on_editor_toggle(s, v, "excluded"),
            inputs=[
                c["application_state"],
                c["scene_gallery_view_toggle"],
            ],
            outputs=[
                c["application_state"],
                c["scene_filter_status"],
                c["scene_gallery"],
                c["propagate_masks_button"],
            ],
        )

        c["sceneundobutton"].click(
            self._undo_last_action,
            inputs=[
                c["application_state"],
                c["scene_gallery_view_toggle"],
            ],
            outputs=[
                c["application_state"],
                c["scene_gallery"],
                c["sceneeditorstatusmd"],
            ],
        )
        
        # State change handler? app_state change trigger?
        # c["scenes_state"].change ...
        # If we change app_state, do we need to trigger updates?
        # Gradio doesn't shallow diff app_state well.
        # But we are manually returning updates from handlers.
        # The only thing missing is: if something ELSE updates app_state.scenes (e.g. AnalysisPipeline),
        # how does the gallery update?
        # In current arch, AnalysisPipeline updates `scenes_state`.
        # We need to make sure AnalysisPipeline updates `app_state.scenes` and then triggers an update.
        # But that's usually done via event returns.
        
        # Dropping c["scenes_state"].change because we handle updates explicitly in handlers.
        
        # New Subject Selection Gallery Handler
        def on_subject_gallery_select(evt: gr.SelectData):
            return str(evt.index + 1)

        c["subject_selection_gallery"].select(on_subject_gallery_select, None, c["scene_editor_subject_id"])

        for comp in [
            c["scene_mask_area_min_input"],
            c["scene_face_sim_min_input"],
            c["scene_quality_score_min_input"],
        ]:
            comp.release(
                self.on_apply_bulk_scene_filters_extended,
                [
                    c["application_state"],
                    c["scene_mask_area_min_input"],
                    c["scene_face_sim_min_input"],
                    c["scene_quality_score_min_input"],
                    c["enable_face_filter_input"],
                    c["scene_gallery_view_toggle"],
                ],
                [
                    c["application_state"],
                    c["scene_filter_status"],
                    c["scene_gallery"],
                    c["propagate_masks_button"],
                ],
            )

        # Gallery size controls
        def update_gallery_layout(columns, height, current_gallery):
            return gr.Gallery(columns=int(columns), height=int(height), value=current_gallery)

        c["scene_gallery_columns"].release(
            update_gallery_layout,
            [c["scene_gallery_columns"], c["scene_gallery_height"], c["scene_gallery"]],
            [c["scene_gallery"]],
        )
        c["scene_gallery_height"].release(
            update_gallery_layout,
            [c["scene_gallery_columns"], c["scene_gallery_height"], c["scene_gallery"]],
            [c["scene_gallery"]],
        )

    def _undo_last_action(self, app_state: ApplicationState, view: str) -> tuple:
        """Reverts the last action by popping from the history stack."""
        prev_scenes = app_state.pop_history()
        
        if prev_scenes is None:
            # Return current state unchanged
            items, index_map, _ = build_scene_gallery_items(
                app_state.scenes, view, app_state.extracted_frames_dir, config=self.config
            )
            return app_state, gr.update(value=items), "Nothing to undo."

        # Restore state
        app_state.scenes = prev_scenes
        save_scene_seeds([Scene(**s) for s in prev_scenes], app_state.extracted_frames_dir, self.logger)
        
        gallery_items, index_map, _ = build_scene_gallery_items(
            prev_scenes, view, app_state.extracted_frames_dir, config=self.config
        )
        status_text, button_update = get_scene_status_text([Scene(**s) for s in prev_scenes])

        return app_state, gr.update(value=gallery_items), "Undid last action."

    def on_reset_scene_wrapper(self, app_state: ApplicationState, view: str, *ana_args):
        """Resets a scene's manual overrides to its initial state."""
        try:
            app_state.push_history(app_state.scenes)
            shot_id = app_state.selected_scene_id
            outdir = app_state.analysis_output_dir
            
            scene_idx = next((i for i, s in enumerate(app_state.scenes) if s["shot_id"] == shot_id), None)
            if scene_idx is None:
                return app_state, gr.update(), "Scene not found."

            scene = app_state.scenes[scene_idx]
            scene.update(
                {
                    "seed_config": {},
                    "seed_result": {},
                    "seed_metrics": {},
                    "manual_status_change": False,
                    "status": "included",
                    "is_overridden": False,
                    "selected_bbox": scene.get("initial_bbox"),
                }
            )
            masker = _create_analysis_context(
                self.config,
                self.logger,
                self.thumbnail_manager,
                self.app.cuda_available,
                self.app.ana_ui_map_keys,
                list(ana_args),
                self.model_registry,
            )
            scene_state = SceneState(app_state.scenes[scene_idx])
            _recompute_single_preview(scene_state, masker, {}, self.thumbnail_manager, self.logger)
            app_state.scenes[scene_idx] = scene_state.data
            save_scene_seeds([Scene(**s) for s in app_state.scenes], outdir, self.logger)
            
            gallery_items, index_map, _ = build_scene_gallery_items(
                app_state.scenes, view, outdir, config=self.config
            )
            return (
                app_state,
                gr.update(value=gallery_items),
                f"Scene {shot_id} reset.",
            )
        except Exception as e:
            self.logger.error(f"Failed to reset scene {shot_id}", exc_info=True)
            return app_state, gr.update(), f"Error: {e}"

    def on_select_for_edit(self, app_state: ApplicationState, view: str, event: Optional[gr.SelectData] = None):
        """Handles selection of a scene from the gallery for editing."""
        sel_idx = getattr(event, "index", None) if event else None
        
        # Guard: invalid selection
        if sel_idx is None or not app_state.scenes:
            return (
                app_state,
                "Status",
                gr.update(),
                "Select a scene.",
                "",
                gr.update(visible=False),
                gr.update(value=[]),
                gr.update(),
                gr.update(),
            )

        # Retrieve scene from mapped index
        if sel_idx >= len(app_state.scene_gallery_index_map):
            self.logger.warning(f"Selection index {sel_idx} out of range for map (len {len(app_state.scene_gallery_index_map)})")
            # If map is stale, fallback or return empty? Return empty for safety
            return (app_state, *[gr.update() for _ in range(8)]) # Sloppy but safe

        scene_idx_in_state = app_state.scene_gallery_index_map[sel_idx]
        scene = app_state.scenes[scene_idx_in_state]
        shotid = scene.get("shot_id")
        
        # Update selected ID in state
        app_state.selected_scene_id = shotid
        
        previews_dir = Path(app_state.analysis_output_dir) / "previews"
        thumb_path = previews_dir / f"scene_{shotid:05d}.jpg"
        gallery_image = self.thumbnail_manager.get(thumb_path) if thumb_path.exists() else None
        gallery_shape = gallery_image.shape[:2] if gallery_image is not None else None
        
        # Update image state
        app_state.gallery_image = gallery_image
        app_state.gallery_shape = gallery_shape

        status_md = f"**Scene {shotid}** (Frames {scene.get('start_frame')}-{scene.get('end_frame')})"
        prompt = (scene.get("seed_config") or {}).get("text_prompt", "")

        # Create Subject Crops for Mini-Gallery
        subject_crops = []
        if gallery_image is not None:
            detections = scene.get("person_detections", [])
            h, w, _ = gallery_image.shape
            for i, det in enumerate(detections):
                bbox = det["bbox"]
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                crop = gallery_image[y1:y2, x1:x2]
                subject_crops.append((crop, f"Subject {i + 1}"))

        return (
            app_state,
            get_scene_status_text([Scene(**s) for s in app_state.scenes])[0],
            gr.update(),
            gr.update(value=status_md),
            gr.update(value=prompt),
            gr.update(visible=True),
            gr.update(value=subject_crops),
            get_scene_status_text([Scene(**s) for s in app_state.scenes])[1],
            gr.update(value=gallery_image),
        )

    def on_editor_toggle(self, app_state: ApplicationState, view: str, new_status: str):
        """Toggles the included/excluded status of a scene."""
        app_state.push_history(app_state.scenes)
        
        selected_shotid = app_state.selected_scene_id
        outputfolder = app_state.extracted_frames_dir
        
        scenes_objs = [Scene(**s) for s in app_state.scenes]
        scenes_objs, status_text, _, button_update = toggle_scene_status(
            scenes_objs, selected_shotid, new_status, outputfolder, self.logger
        )
        app_state.scenes = [s.model_dump() for s in scenes_objs]
        
        items, index_map, _ = build_scene_gallery_items(
            app_state.scenes, view, outputfolder, config=self.config
        )
        return app_state, status_text, gr.update(value=items), button_update

    def on_apply_bulk_scene_filters_extended(
        self, app_state: ApplicationState, min_mask_pct, min_face_sim, min_quality, enable_face_filter, view
    ):
        """Applies bulk filters to scenes based on metric thresholds."""
        app_state.push_history(app_state.scenes)
        
        output_dir = app_state.extracted_frames_dir
        
        changed_count = 0
        min_mask_pct = float(min_mask_pct) if min_mask_pct is not None else 0.0
        min_face_sim = float(min_face_sim) if min_face_sim is not None else 0.0
        min_quality = float(min_quality) if min_quality is not None else 0.0

        for scene in app_state.scenes:
            if scene.get("is_overridden", False) or scene.get("manual_status_change", False):
                continue

            current_status = scene.get("status", "included")
            should_exclude = False
            reason = []

            # Check Subj Area
            if min_mask_pct > 0:
                area = (scene.get("seed_result") or {}).get("mask_area_pct", 0)
                if area < min_mask_pct:
                    should_exclude = True
                    reason.append(f"Area {area:.1f}% < {min_mask_pct}%")

            # Check Face Sim (only if enabled globally AND logic enabled)
            if enable_face_filter and min_face_sim > 0:
                sim = (scene.get("seed_result") or {}).get("face_sim", 0)
                if sim < min_face_sim:
                    should_exclude = True
                    reason.append(f"Face {sim:.2f} < {min_face_sim}")

            # Check Quality
            if min_quality > 0:
                q = (scene.get("seed_metrics") or {}).get("quality_score", 0)
                if q < min_quality:
                    should_exclude = True
                    reason.append(f"Quality {q:.1f} < {min_quality}")

            new_status = "excluded" if should_exclude else "included"

            if current_status != new_status or scene.get("rejection_reasons") != reason:
                scene["status"] = new_status
                scene["rejection_reasons"] = reason
                changed_count += 1

        save_scene_seeds([Scene(**s) for s in app_state.scenes], output_dir, self.logger)
        items, index_map, _ = build_scene_gallery_items(
            app_state.scenes, view, output_dir, config=self.config
        )
        status_text, button_update = get_scene_status_text([Scene(**s) for s in app_state.scenes])

        app_state.scene_gallery_index_map = index_map
        return (
            app_state,
            status_text,
            gr.update(value=items),
            button_update,
        )
