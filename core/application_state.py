from __future__ import annotations
import copy
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class ApplicationState(BaseModel):
    """Consolidated state model for the application."""
    extracted_video_path: str = ""
    extracted_frames_dir: str = ""
    analysis_output_dir: str = ""
    analysis_metadata_path: str = ""
    all_frames_data: List[dict] = Field(default_factory=list)
    per_metric_values: Dict[str, List[float]] = Field(default_factory=dict)
    scenes: List[dict] = Field(default_factory=list)
    selected_scene_id: Optional[int] = None
    scene_gallery_index_map: List[int] = Field(default_factory=list)
    gallery_image: Optional[Any] = None
    gallery_shape: Optional[Any] = None
    discovered_faces: List[dict] = Field(default_factory=list)
    resume: bool = False
    enable_subject_mask: bool = True
    min_mask_area_pct: float = 1.0
    sharpness_base_scale: float = 2500.0
    edge_strength_base_scale: float = 100.0
    smart_filter_enabled: bool = False
    scene_history: List[List[dict]] = Field(default_factory=list)

    def push_history(self, scenes: List[dict]):
        """Push a snapshot of scenes to history."""
        # Deep copy to ensure isolation
        self.scene_history.append(copy.deepcopy(scenes))
        if len(self.scene_history) > 10:
            self.scene_history.pop(0)

    def pop_history(self) -> Optional[List[dict]]:
        """Pop the last snapshot from history."""
        if not self.scene_history:
            return None
        return self.scene_history.pop()
