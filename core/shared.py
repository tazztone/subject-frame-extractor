"""
Shared utilities for Frame Extractor & Analyzer

This module contains pure functions that are shared between core and UI modules,
resolving circular import issues.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from core.enums import SceneStatus

if TYPE_CHECKING:
    from core.config import Config
    from core.models import Scene

logger = logging.getLogger(__name__)


# TODO: Add caching for repeated scene status checks
def scene_matches_view(scene: "Scene", view: str) -> bool:
    """
    Check if a scene matches the specified view filter.

    Args:
        scene: Scene object to check
        view: One of "All", "Kept", or "Rejected"

    Returns:
        True if the scene matches the view filter
    """
    status = getattr(scene, "status", SceneStatus.INCLUDED)
    if view == "All":
        return status in (SceneStatus.INCLUDED, SceneStatus.EXCLUDED, SceneStatus.PENDING)
    if view == "Kept":
        return status == SceneStatus.INCLUDED
    if view == "Rejected":
        return status == SceneStatus.EXCLUDED
    return False


def create_scene_thumbnail_with_badge(
    thumb_img: np.ndarray,
    scene_idx: int,
    status: Union[bool, str, "SceneStatus"],
    config: Optional["Config"] = None,
    use_svg: bool = True,
) -> Union[np.ndarray, str]:
    """
    Create a scene thumbnail with a visual badge indicating status.

    Args:
        thumb_img: RGB thumbnail image
        scene_idx: Index of the scene
        status: The status of the scene (bool for legacy is_excluded, or string like 'excluded', 'pending', 'error')
        config: Application configuration (optional)
        use_svg: Whether to return an SVG data URL for better scaling (default: True)

    Returns:
        Thumbnail with badge overlay (numpy array or SVG data URL string)
    """
    h, w = thumb_img.shape[:2]

    # Normalize status
    if isinstance(status, bool):
        status_str = "excluded" if status else "included"
    elif hasattr(status, "value"):
        status_str = str(status.value).lower()
    else:
        status_str = str(status).lower()

    if status_str == "included":
        return thumb_img

    # Defaults for excluded
    border_color = (33, 128, 141)  # Teal-ish color
    text_color = (255, 255, 255)  # White
    badge_char = "E"

    if status_str == "excluded":
        if config:
            border_color = tuple(config.visualization_badge_excluded_color)
            text_color = tuple(config.visualization_badge_text_color)
        badge_char = "E"
    elif status_str == "pending":
        border_color = (128, 128, 128)  # Gray
        badge_char = "P"
    elif status_str == "error":
        border_color = (255, 0, 0)  # Red
        badge_char = "!"
    else:
        return thumb_img

    if use_svg:
        import base64

        # Convert image to base64
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(thumb_img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        img_b64 = base64.b64encode(buffer).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{img_b64}"

        svg_color = f"rgb({border_color[0]}, {border_color[1]}, {border_color[2]})"
        svg_text_color = f"rgb({text_color[0]}, {text_color[1]}, {text_color[2]})"

        badge_size = int(min(w, h) * 0.15)
        # Circle center (top-right)
        cx = w - badge_size // 2 - 5
        cy = badge_size // 2 + 5

        # Create SVG string
        svg = f"""<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">
            <image href="{data_url}" width="{w}" height="{h}" />
            <rect x="2" y="2" width="{w-4}" height="{h-4}" fill="none" stroke="{svg_color}" stroke-width="4" />
            <circle cx="{cx}" cy="{cy}" r="{badge_size // 2}" fill="{svg_color}" />
            <text x="{cx}" y="{cy}" font-family="sans-serif" font-size="{int(badge_size * 0.7)}" fill="{svg_text_color}" text-anchor="middle" dominant-baseline="central">{badge_char}</text>
        </svg>"""

        svg_b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
        return f"data:image/svg+xml;base64,{svg_b64}"

    # Raster fallback
    thumb = thumb_img.copy()
    cv2.rectangle(thumb, (0, 0), (w - 1, h - 1), border_color, 4)
    badge_size = int(min(w, h) * 0.15)
    badge_pos = (w - badge_size - 5, 5)

    # Draw filled circle with border color
    cv2.circle(
        thumb,
        (badge_pos[0] + badge_size // 2, badge_pos[1] + badge_size // 2),
        badge_size // 2,
        border_color,
        -1,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(badge_char, font, 0.5, 2)[0]

    # Center text inside the circle
    text_x = badge_pos[0] + (badge_size - text_size[0]) // 2
    text_y = badge_pos[1] + (badge_size + text_size[1]) // 2

    cv2.putText(thumb, badge_char, (text_x, text_y), font, 0.5, text_color, 2)
    return thumb


def scene_caption(scene: Union[dict, "Scene"]) -> str:
    """
    Generate a caption string for a scene.

    Args:
        scene: Scene object or dict

    Returns:
        Caption string with scene ID, frame range, and status
    """
    if isinstance(scene, dict):
        shot = scene.get("shot_id", 0)
        start, end = scene.get("start_frame", 0), scene.get("end_frame", 0)
        status = scene.get("status", SceneStatus.INCLUDED)
        rejection_reasons = scene.get("rejection_reasons", [])
        seed_type = scene.get("seed_type")
    else:
        shot = scene.shot_id
        start, end = scene.start_frame, scene.end_frame
        status = scene.status
        rejection_reasons = scene.rejection_reasons or []
        seed_type = scene.seed_type

    status_icon = "✅" if status == SceneStatus.INCLUDED else "❌"
    caption = f"Scene {shot} [{start}-{end}] {status_icon}"
    if status == SceneStatus.EXCLUDED and rejection_reasons:
        caption += f"\n({', '.join(rejection_reasons)})"
    if seed_type:
        caption += f"\nSeed: {seed_type}"
    return caption


def build_scene_gallery_items(
    scenes: Sequence[Union[dict, "Scene"]],
    view: str,
    output_dir: str,
    page_num: int = 1,
    page_size: int = 20,
    config: Optional["Config"] = None,
) -> Tuple[List[Tuple], List[int], int]:
    """
    Build gallery items for scene display.

    This function is moved from ui/gallery_utils.py to break the circular
    import between core/pipelines.py and ui/gallery_utils.py.

    Args:
        scenes: List of Scene objects or dicts
        view: View filter ("All", "Kept", "Rejected")
        output_dir: Path to output directory
        page_num: Current page number (1-indexed)
        page_size: Items per page
        config: Application configuration (optional)

    Returns:
        Tuple of (gallery_items, index_map, total_pages)
    """
    from core.models import Scene as SceneModel

    items: List[Tuple[Optional[Union[np.ndarray, str]], str]] = []
    index_map: List[int] = []

    if not scenes:
        return [], [], 1

    # Ensure scenes are Scene objects
    scenes_objs = []
    for s in scenes:
        if isinstance(s, dict):
            scenes_objs.append(SceneModel(**s))
        else:
            scenes_objs.append(s)

    previews_dir = Path(output_dir) / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    # Filter scenes based on view
    filtered_scenes = [(i, s) for i, s in enumerate(scenes_objs) if scene_matches_view(s, view)]

    # Pagination
    total_pages = max(1, (len(filtered_scenes) + page_size - 1) // page_size)
    start_idx = (page_num - 1) * page_size
    end_idx = start_idx + page_size
    page_scenes = filtered_scenes[start_idx:end_idx]

    for i, s in page_scenes:
        # Resolve preview: prefer stored path, then glob for timestamped files
        thumb_path = None
        if hasattr(s, "preview_path") and s.preview_path:
            p = Path(s.preview_path)
            if p.exists():
                thumb_path = p

        if thumb_path is None:
            # Glob fallback for timestamped files (e.g. scene_00000_1775129777.jpg)
            matches = sorted(previews_dir.glob(f"scene_{s.shot_id:05d}*.jpg"))
            thumb_path = matches[-1] if matches else None

        if thumb_path is None:
            continue

        try:
            thumb_img_np = cv2.imread(str(thumb_path))
            if thumb_img_np is None:
                continue
            thumb_img_np = cv2.cvtColor(thumb_img_np, cv2.COLOR_BGR2RGB)
            badged_thumb = create_scene_thumbnail_with_badge(thumb_img_np, i, s.status, config=config)
            items.append((badged_thumb, scene_caption(s)))
        except Exception as e:
            logger.error(f"Failed to create thumbnail for scene {getattr(s, 'shot_id', 'unknown')} at {thumb_path}: {e}", exc_info=True)
            continue
        index_map.append(i)

    return items, index_map, total_pages
