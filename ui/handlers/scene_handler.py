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

        def on_page_change(scenes, view, output_dir, page_num):
            """Handle page change - returns gallery items and dropdown update with choices."""
            try:
                current_page = int(page_num) if page_num else 1
            except (ValueError, TypeError):
                current_page = 1
            items, index_map, total_pages = build_scene_gallery_items(scenes, view, output_dir, page_num=current_page)
            # Generate page choices for dropdown
            page_choices = [str(i) for i in range(1, total_pages + 1)] if total_pages > 0 else ["1"]
            return (
                gr.update(value=items),
                index_map,
                f"/ {total_pages} pages",
                gr.update(choices=page_choices, value=str(current_page)),
            )

        def on_view_change(scenes, view, output_dir):
            """Handle view filter change - reset to page 1 and update dropdown choices."""
            items, index_map, total_pages = build_scene_gallery_items(scenes, view, output_dir, page_num=1)
            page_choices = [str(i) for i in range(1, total_pages + 1)] if total_pages > 0 else ["1"]
            return items, index_map, f"/ {total_pages} pages", gr.update(choices=page_choices, value="1")

        def on_next_page(scenes, view, output_dir, page_num):
            """Go to next page (clamped to max)."""
            try:
                current = int(page_num) if page_num else 1
            except (ValueError, TypeError):
                current = 1
            # Get total pages to clamp
            _, _, total_pages = build_scene_gallery_items(scenes, view, output_dir, page_num=1)
            new_page = min(current + 1, total_pages)
            return on_page_change(scenes, view, output_dir, new_page)

        def on_prev_page(scenes, view, output_dir, page_num):
            """Go to previous page."""
            try:
                current = int(page_num) if page_num else 1
            except (ValueError, TypeError):
                current = 1
            return on_page_change(scenes, view, output_dir, max(1, current - 1))

        # --- Wire existing components ---
        # Note: We rely on self.app.components having these keys populated by SceneTabBuilder

        c["scene_gallery_view_toggle"].change(
            on_view_change,
            [c["scenes_state"], c["scene_gallery_view_toggle"], c["extracted_frames_dir_state"]],
            [c["scene_gallery"], c["scene_gallery_index_map_state"], c["total_pages_label"], c["page_number_input"]],
        )
        c["next_page_button"].click(
            on_next_page,
            [
                c["scenes_state"],
                c["scene_gallery_view_toggle"],
                c["extracted_frames_dir_state"],
                c["page_number_input"],
            ],
            [c["scene_gallery"], c["scene_gallery_index_map_state"], c["total_pages_label"], c["page_number_input"]],
        )
        c["prev_page_button"].click(
            on_prev_page,
            [
                c["scenes_state"],
                c["scene_gallery_view_toggle"],
                c["extracted_frames_dir_state"],
                c["page_number_input"],
            ],
            [c["scene_gallery"], c["scene_gallery_index_map_state"], c["total_pages_label"], c["page_number_input"]],
        )
        c["page_number_input"].change(
            on_page_change,
            [
                c["scenes_state"],
                c["scene_gallery_view_toggle"],
                c["extracted_frames_dir_state"],
                c["page_number_input"],
            ],
            [c["scene_gallery"], c["scene_gallery_index_map_state"], c["total_pages_label"], c["page_number_input"]],
        )

        c["scene_gallery"].select(
            self.on_select_for_edit,
            inputs=[
                c["scenes_state"],
                c["scene_gallery_view_toggle"],
                c["scene_gallery_index_map_state"],
                c["extracted_frames_dir_state"],
            ],
            outputs=[
                c["scenes_state"],
                c["scene_filter_status"],
                c["scene_gallery"],
                c["scene_gallery_index_map_state"],
                c["selected_scene_id_state"],
                c["sceneeditorstatusmd"],
                c["sceneeditorpromptinput"],
                c["scene_editor_group"],
                c["gallery_image_state"],
                c["gallery_shape_state"],
                c["subject_selection_gallery"],
                c["propagate_masks_button"],
                c["gallery_image_preview"],
            ],
        )

        c["scenerecomputebutton"].click(
            fn=lambda scenes, shot_id, outdir, view, txt, history, *ana_args: _wire_recompute_handler(
                self.config,
                self.logger,
                self.thumbnail_manager,
                [Scene(**s) for s in scenes],
                shot_id,
                outdir,
                txt,
                view,
                self.app.ana_ui_map_keys,
                list(ana_args),
                self.app.cuda_available,
                self.model_registry,
            ),
            inputs=[
                c["scenes_state"],
                c["selected_scene_id_state"],
                c["analysis_output_dir_state"],
                c["scene_gallery_view_toggle"],
                c["sceneeditorpromptinput"],
                c["scene_history_state"],
                *self.app.ana_input_components,
            ],
            outputs=[
                c["scenes_state"],
                c["scene_gallery"],
                c["scene_gallery_index_map_state"],
                c["sceneeditorstatusmd"],
                c["scene_history_state"],
            ],
        )

        c["sceneresetbutton"].click(
            self.on_reset_scene_wrapper,
            inputs=[
                c["scenes_state"],
                c["selected_scene_id_state"],
                c["analysis_output_dir_state"],
                c["scene_gallery_view_toggle"],
                c["scene_history_state"],
            ]
            + self.app.ana_input_components,
            outputs=[
                c["scenes_state"],
                c["scene_gallery"],
                c["scene_gallery_index_map_state"],
                c["sceneeditorstatusmd"],
                c["scene_history_state"],
            ],
        )

        c["sceneincludebutton"].click(
            lambda s, sid, out, v, h: self.on_editor_toggle(s, sid, out, v, "included", h),
            inputs=[
                c["scenes_state"],
                c["selected_scene_id_state"],
                c["extracted_frames_dir_state"],
                c["scene_gallery_view_toggle"],
                c["scene_history_state"],
            ],
            outputs=[
                c["scenes_state"],
                c["scene_filter_status"],
                c["scene_gallery"],
                c["scene_gallery_index_map_state"],
                c["propagate_masks_button"],
                c["scene_history_state"],
            ],
        )
        c["sceneexcludebutton"].click(
            lambda s, sid, out, v, h: self.on_editor_toggle(s, sid, out, v, "excluded", h),
            inputs=[
                c["scenes_state"],
                c["selected_scene_id_state"],
                c["extracted_frames_dir_state"],
                c["scene_gallery_view_toggle"],
                c["scene_history_state"],
            ],
            outputs=[
                c["scenes_state"],
                c["scene_filter_status"],
                c["scene_gallery"],
                c["scene_gallery_index_map_state"],
                c["propagate_masks_button"],
                c["scene_history_state"],
            ],
        )

        c["sceneundobutton"].click(
            self._undo_last_action,
            inputs=[
                c["scenes_state"],
                c["scene_history_state"],
                c["extracted_frames_dir_state"],
                c["scene_gallery_view_toggle"],
            ],
            outputs=[
                c["scenes_state"],
                c["scene_gallery"],
                c["scene_gallery_index_map_state"],
                c["sceneeditorstatusmd"],
                c["scene_history_state"],
            ],
        )
        c["scenes_state"].change(
            lambda s, v, o: (build_scene_gallery_items(s, v, o)[0], build_scene_gallery_items(s, v, o)[1]),
            [c["scenes_state"], c["scene_gallery_view_toggle"], c["extracted_frames_dir_state"]],
            [c["scene_gallery"], c["scene_gallery_index_map_state"]],
        )

        # New Subject Selection Gallery Handler
        def on_subject_gallery_select(evt: gr.SelectData):
            # Map index to radio value (index + 1 as string) and trigger the hidden radio change
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
                    c["scenes_state"],
                    c["scene_mask_area_min_input"],
                    c["scene_face_sim_min_input"],
                    c["scene_quality_score_min_input"],
                    c["enable_face_filter_input"],
                    c["extracted_frames_dir_state"],
                    c["scene_gallery_view_toggle"],
                    c["scene_history_state"],
                ],
                [
                    c["scenes_state"],
                    c["scene_filter_status"],
                    c["scene_gallery"],
                    c["scene_gallery_index_map_state"],
                    c["propagate_masks_button"],
                    c["scene_history_state"],
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

    def _push_history(self, scenes: List[Dict], history: Deque) -> Deque:
        """Pushes the current scene state to the history stack for undo support."""
        history.append(copy.deepcopy(scenes))
        return history

    def _undo_last_action(self, scenes: List[Dict], history: Deque, output_dir: str, view: str) -> tuple:
        """Reverts the last action by popping from the history stack."""
        if not history:
            return scenes, gr.update(), gr.update(), "Nothing to undo.", history

        prev_scenes = history.pop()
        save_scene_seeds([Scene(**s) for s in prev_scenes], output_dir, self.logger)
        gallery_items, index_map, _ = build_scene_gallery_items(prev_scenes, view, output_dir)
        status_text, button_update = get_scene_status_text([Scene(**s) for s in prev_scenes])

        return prev_scenes, gr.update(value=gallery_items), gr.update(value=index_map), "Undid last action.", history

    def on_reset_scene_wrapper(self, scenes, shot_id, outdir, view, history, *ana_args):
        """Resets a scene's manual overrides to its initial state."""
        try:
            history = self._push_history(scenes, history)
            scene_idx = next((i for i, s in enumerate(scenes) if s["shot_id"] == shot_id), None)
            if scene_idx is None:
                return scenes, gr.update(), gr.update(), "Scene not found.", history
            scene = scenes[scene_idx]
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
            scene_state = SceneState(scenes[scene_idx])
            _recompute_single_preview(scene_state, masker, {}, self.thumbnail_manager, self.logger)
            scenes[scene_idx] = scene_state.data
            save_scene_seeds([Scene(**s) for s in scenes], outdir, self.logger)
            gallery_items, index_map, _ = build_scene_gallery_items(scenes, view, outdir)
            return (
                scenes,
                gr.update(value=gallery_items),
                gr.update(value=index_map),
                f"Scene {shot_id} reset.",
                history,
            )
        except Exception as e:
            self.logger.error(f"Failed to reset scene {shot_id}", exc_info=True)
            return scenes, gr.update(), gr.update(), f"Error: {e}", history

    def on_select_for_edit(self, scenes, view, indexmap, outputdir, event: Optional[gr.EventData] = None):
        """Handles selection of a scene from the gallery for editing."""
        sel_idx = getattr(event, "index", None) if event else None
        if sel_idx is None or not scenes:
            return (
                scenes,
                "Status",
                gr.update(),
                indexmap,
                None,
                "Select a scene.",
                "",
                gr.update(visible=False),
                None,
                None,
                gr.update(value=[]),
                gr.update(),
                gr.update(),
            )

        scene_idx_in_state = indexmap[sel_idx]
        scene = scenes[scene_idx_in_state]
        shotid = scene.get("shot_id")
        previews_dir = Path(outputdir) / "previews"
        thumb_path = previews_dir / f"scene_{shotid:05d}.jpg"
        gallery_image = self.thumbnail_manager.get(thumb_path) if thumb_path.exists() else None
        gallery_shape = gallery_image.shape[:2] if gallery_image is not None else None

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
            scenes,
            get_scene_status_text([Scene(**s) for s in scenes])[0],
            gr.update(),
            indexmap,
            shotid,
            gr.update(value=status_md),
            gr.update(value=prompt),
            gr.update(visible=True),
            gallery_image,
            gallery_shape,
            gr.update(value=subject_crops),
            get_scene_status_text([Scene(**s) for s in scenes])[1],
            gr.update(value=gallery_image),
        )

    def on_editor_toggle(self, scenes, selected_shotid, outputfolder, view, new_status, history):
        """Toggles the included/excluded status of a scene."""
        history = self._push_history(scenes, history)
        scenes_objs = [Scene(**s) for s in scenes]
        scenes_objs, status_text, _, button_update = toggle_scene_status(
            scenes_objs, selected_shotid, new_status, outputfolder, self.logger
        )
        scenes = [s.model_dump() for s in scenes_objs]
        items, index_map, _ = build_scene_gallery_items(scenes, view, outputfolder)
        return scenes, status_text, gr.update(value=items), gr.update(value=index_map), button_update, history

    def on_apply_bulk_scene_filters_extended(
        self, scenes, min_mask_pct, min_face_sim, min_quality, enable_face_filter, output_dir, view, history
    ):
        """Applies bulk filters to scenes based on metric thresholds."""
        history = self._push_history(scenes, history)
        changed_count = 0
        min_mask_pct = float(min_mask_pct) if min_mask_pct is not None else 0.0
        min_face_sim = float(min_face_sim) if min_face_sim is not None else 0.0
        min_quality = float(min_quality) if min_quality is not None else 0.0

        for scene in scenes:
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

        save_scene_seeds([Scene(**s) for s in scenes], output_dir, self.logger)
        items, index_map, _ = build_scene_gallery_items(scenes, view, output_dir)
        status_text, button_update = get_scene_status_text([Scene(**s) for s in scenes])

        return (
            scenes,
            status_text,
            gr.update(value=items),
            gr.update(value=index_map),
            button_update,
            history,
        )
