from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import gradio as gr
import numpy as np

from core.application_state import ApplicationState
from core.context import AnalysisContext
from core.face_clustering import cluster_faces, get_cluster_representative, scan_faces_in_session
from ui.decorators import safe_ui_callback

if TYPE_CHECKING:
    from ui.app_ui import AppUI


class SubjectHandler:
    """
    Handles subject discovery and face clustering callbacks.
    Extracted from monolith app_ui.py to improve maintainability.
    """

    def __init__(self, app: "AppUI"):
        self.app = app
        self.config = app.config
        self.logger = app.logger
        self.thumbnail_manager = app.thumbnail_manager
        self.model_registry = app.model_registry
        self.gallery_to_cluster_map: Dict[int, int] = {}

    @safe_ui_callback("Face Clustering")
    def on_identity_confidence_change(self, confidence: float, state: ApplicationState) -> Any:
        """Updates the face discovery gallery based on clustering confidence."""
        all_faces = state.discovered_faces
        if not all_faces:
            return gr.update(value=[])

        labels, cluster_map = cluster_faces(all_faces, confidence)
        self.gallery_to_cluster_map = cluster_map

        gallery_items = []
        for idx, label in cluster_map.items():
            cluster_faces_list = [all_faces[i] for i, l in enumerate(labels) if l == label]
            best_face = max(cluster_faces_list, key=lambda x: x["det_score"])

            thumb_rgb = self.thumbnail_manager.get(Path(best_face["thumb_path"]))
            if thumb_rgb is not None:
                x1, y1, x2, y2 = best_face["bbox"].astype(int)
                face_crop = thumb_rgb[y1:y2, x1:x2]
                gallery_items.append((face_crop, f"Person {label}"))

        return gr.update(value=gallery_items)

    @safe_ui_callback("Face Selection")
    def on_discovered_face_select(
        self, state: ApplicationState, confidence: float, evt: Optional[gr.SelectData] = None
    ) -> tuple[Optional[str], Optional[np.ndarray], str]:
        """Handles selection of a face cluster from the discovery gallery."""
        all_faces = state.discovered_faces
        if not all_faces or evt is None or evt.index is None:
            return "", None, "⚠️ Selection Failed"

        selected_label = self.gallery_to_cluster_map.get(evt.index)
        if selected_label is None:
            return "", None, "Selection Failed"

        labels, _ = cluster_faces(all_faces, confidence)

        return get_cluster_representative(
            all_faces, labels, selected_label, state.extracted_video_path, state.extracted_frames_dir
        )

    @safe_ui_callback("Subject Discovery")
    def on_find_subjects_from_video(self, current_state: ApplicationState, *args) -> dict:
        """Scans the video for subjects to populate the discovery gallery.

        Returns: Dictionary of component updates.
        """
        # Assert list lengths match to prevent silent crashes
        assert len(args) == len(self.app.ana_ui_map_keys), (
            f"Expected {len(self.app.ana_ui_map_keys)} arguments, got {len(args)}"
        )

        new_state = current_state.model_copy()
        c = self.app.components
        self.logger.info("Scan Video for Subjects clicked")
        params = self.app._create_pre_analysis_event(current_state, *args)
        output_dir = Path(params.output_folder)
        self.logger.info(f"Output dir: {output_dir}, exists: {output_dir.exists()}")
        if not output_dir.exists():
            self.logger.warning("Output directory does not exist - run extraction first")
            msg = "**Run extraction first** - No video frames found."
            return {
                c["unified_status"]: "**Face Discovery Failed.** Run extraction first.",
                c["find_people_status"]: msg,
                c["discovered_people_group"]: gr.update(visible=False),
                c["discovered_faces_gallery"]: [],
                c["identity_confidence_slider"]: 0.5,
                c["application_state"]: new_state,
            }

        # Create context to pass
        context = AnalysisContext(
            config=self.config,
            logger=self.logger,
            progress_queue=self.app.progress_queue,
            cancel_event=self.app.cancel_event,
            thumbnail_manager=self.thumbnail_manager,
            model_registry=self.model_registry,
            cuda_available=self.app.cuda_available,
        )

        all_faces = scan_faces_in_session(context, params)

        self.logger.info(f"Found {len(all_faces)} faces in video")
        new_state.discovered_faces = all_faces

        if not all_faces:
            self.logger.info("No faces found in sampled frames")
            msg = "**No faces detected** in sampled frames. Try adjusting sample rate."
            return {
                c["unified_status"]: "**Face Discovery Finished.** No faces found.",
                c["find_people_status"]: msg,
                c["discovered_people_group"]: gr.update(visible=False),
                c["discovered_faces_gallery"]: [],
                c["identity_confidence_slider"]: 0.5,
                c["application_state"]: new_state,
            }

        # Get clustered faces for gallery
        gallery_items = self.on_identity_confidence_change(0.5, new_state)
        n_people = len(self.gallery_to_cluster_map)
        success_msg = f"Found **{n_people} unique people** from {len(all_faces)} face detections."
        return {
            c["unified_status"]: success_msg,
            c["find_people_status"]: success_msg,
            c["discovered_people_group"]: gr.update(visible=True),
            c["discovered_faces_gallery"]: gallery_items,
            c["identity_confidence_slider"]: 0.5,
            c["application_state"]: new_state,
        }
