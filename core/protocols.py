"""
Protocol definitions for dependency injection and static type checking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Protocol, Tuple

if TYPE_CHECKING:
    import numpy as np

    from core.models import Frame, Scene


class DatabaseProtocol(Protocol):
    """Interface for the Application Database."""

    def save_frame(self, frame_id: str, scene_id: str, frame_data: Frame) -> bool: ...
    def get_frame(self, frame_id: str) -> Optional[Frame]: ...
    def save_scene(self, scene_id: str, scene_data: Scene) -> bool: ...
    def get_scene(self, scene_id: str) -> Optional[Scene]: ...
    def update_scene_status(self, scene_id: str, status: str) -> bool: ...
    def execute(self, query: str, params: Optional[Tuple] = None) -> Any: ...


class ThumbnailManagerProtocol(Protocol):
    """Interface for the Thumbnail Manager."""

    def get(self, frame_id: str) -> Optional[np.ndarray]: ...
    def save(self, frame_id: str, image: np.ndarray) -> bool: ...
    def clear_cache(self) -> None: ...
