from __future__ import annotations

import gradio as gr
from typing import TYPE_CHECKING, List, Dict, Any
from pathlib import Path
from core.photo_utils import ingest_folder

if TYPE_CHECKING:
    from ui.app_ui import AppUI

class PhotoTabBuilder:
    """Builds the 'Photo Culling' tab."""

    def __init__(self, app: "AppUI"):
        self.app = app

    def build(self):
        with gr.Row():
            with gr.Column(scale=3):
                self._build_gallery_area()
            with gr.Column(scale=1):
                self._build_sidebar()

    def _build_sidebar(self):
        gr.Markdown("### 📥 Import Photos")
        
        folder_input = gr.Textbox(
            label="Folder Path",
            placeholder="/path/to/raws",
            info="Supports RAW (CR2, NEF, ARW) and JPEG"
        )
        import_btn = gr.Button("Import Folder", variant="primary")
        
        gr.Markdown("---")
        gr.Markdown("### ⚖️ Scoring Weights")
        
        with gr.Accordion("Configure Quality Scoring", open=True):
            s_sharpness = gr.Slider(0.0, 1.0, value=0.4, label="Sharpness", info="Focus detection using Laplacian variance.")
            s_niqe = gr.Slider(0.0, 1.0, value=0.3, label="Naturalness (NIQE)", info="Perceptual quality model (blind/no-reference).")
            s_face = gr.Slider(0.0, 1.0, value=0.2, label="Face Prominence", info="Detection confidence and face area relative to image.")
            s_entropy = gr.Slider(0.0, 1.0, value=0.1, label="Information (Entropy)", info="Shannon entropy indicating detail/complexity.")
            
            recalc_btn = gr.Button("Recalculate Scores")

        gr.Markdown("---")
        export_btn = gr.Button("Sync XMP Sidecars", variant="secondary")
        
        # Event Handlers
        import_btn.click(
            self._on_import_click,
            inputs=[folder_input, self.app.components["application_state"]],
            outputs=[
                self.app.components["photo_gallery"],
                self.app.components["photo_status"],
                self.app.components["application_state"]
            ]
        )
        
        # Save components for reference
        self.app.components["photo_folder_input"] = folder_input
        self.app.components["photo_import_btn"] = import_btn
        self.app.components["photo_recalc_btn"] = recalc_btn
        self.app.components["photo_export_btn"] = export_btn
        self.app.components["photo_weights"] = [s_sharpness, s_niqe, s_face, s_entropy]

    def _build_gallery_area(self):
        with gr.Row():
            status_label = gr.Markdown("**Ready to import.**", elem_id="photo_status")
            self.app.components["photo_status"] = status_label
            
            with gr.Row():
                prev_btn = gr.Button("◀ Prev", size="sm")
                next_btn = gr.Button("Next ▶", size="sm")

        gallery = gr.Gallery(
            label="Photo Culling",
            show_label=False,
            columns=4,
            rows=3,
            height=800,
            object_fit="contain",
            allow_preview=True,
            elem_id="photo_gallery"
        )
        self.app.components["photo_gallery"] = gallery
        
        # Pagination Handlers
        prev_btn.click(
            self._on_prev_page,
            inputs=[self.app.components["application_state"]],
            outputs=[gallery, status_label, self.app.components["application_state"]]
        )
        next_btn.click(
            self._on_next_page,
            inputs=[self.app.components["application_state"]],
            outputs=[gallery, status_label, self.app.components["application_state"]]
        )

    def _on_import_click(self, folder_path_str: str, app_state):
        if not folder_path_str:
            return gr.update(), "⚠️ Please enter a folder path.", app_state
            
        path = Path(folder_path_str)
        if not path.exists() or not path.is_dir():
            return gr.update(), f"❌ Invalid folder: {path}", app_state
            
        # Ingest
        # We assume output_dir is current dir / "previews" for now? 
        # Or better: create a ".previews" inside the source folder to keep it contained.
        preview_dir = path / ".previews"
        
        photos = ingest_folder(path, preview_dir)
        
        if not photos:
            return gr.update(), f"⚠️ No valid images found in {path}", app_state
            
        app_state.photos = photos
        app_state.photo_page = 0
        
        # Refresh Gallery
        gallery_items, status = self._render_gallery_page(app_state)
        return gallery_items, status, app_state

    def _on_prev_page(self, app_state):
        if app_state.photo_page > 0:
            app_state.photo_page -= 1
        return (*self._render_gallery_page(app_state), app_state)

    def _on_next_page(self, app_state):
        # Calculate max page
        max_page = (len(app_state.photos) - 1) // app_state.photo_page_size
        if app_state.photo_page < max_page:
            app_state.photo_page += 1
        return (*self._render_gallery_page(app_state), app_state)

    def _render_gallery_page(self, app_state):
        start_idx = app_state.photo_page * app_state.photo_page_size
        end_idx = start_idx + app_state.photo_page_size
        page_photos = app_state.photos[start_idx:end_idx]
        
        gallery_items = []
        for p in page_photos:
            # Format: (image_path, label)
            # Label can include score if available
            label = p["id"]
            if "quality_score" in p.get("scores", {}):
                 label += f" | {p['scores']['quality_score']:.1f}"
            
            gallery_items.append((p["preview"], label))
            
        status = f"**Page {app_state.photo_page + 1}** ({start_idx+1}-{min(end_idx, len(app_state.photos))} of {len(app_state.photos)})"
        return gallery_items, status
