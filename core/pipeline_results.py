from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class PipelineResult(BaseModel):
    """Base class for all pipeline results to ensure consistent signaling."""

    success: bool = True
    error: Optional[str] = None
    unified_log: Optional[str] = None
    done: bool = False
    # Backward compatibility fields for UI/Legacy consumers
    status_message: Optional[str] = None
    error_message: Optional[str] = None


class PreAnalysisResult(PipelineResult):
    """Typed result for the pre-analysis pipeline stage."""

    scenes: List[dict] = Field(default_factory=list)
    output_dir: str = ""
    video_path: str = ""
    done: bool = True
    # Signaling results visibility — UI layer interprets this to update components.
    show_results: bool = False


class ExtractionResult(PipelineResult):
    """Typed result for the extraction pipeline stage."""

    extracted_video_path_state: str = ""
    extracted_frames_dir_state: str = ""


class PropagationResult(PipelineResult):
    """Typed result for the mask propagation pipeline stage."""

    output_dir: str = ""
    mask_count: int = 0


class AnalysisResult(PipelineResult):
    """Typed result for the analysis-only pipeline stage."""

    output_dir: str = ""
    metadata_path: str = ""
