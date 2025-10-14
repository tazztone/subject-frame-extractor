"""Frame processing and utility functions."""

import json
import re
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from app.logging_enhanced import EnhancedLogger


def postprocess_mask(mask: np.ndarray, fill_holes: bool = True,
                     keep_largest_only: bool = True) -> np.ndarray:
    """Post-process SAM2 masks."""
    if mask is None or mask.size == 0:
        return mask

    binary_mask = (mask > 128).astype(np.uint8)

    # Fill holes with morphological closing
    if fill_holes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Keep only largest connected component
    if keep_largest_only:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            binary_mask = (labels == largest_label).astype(np.uint8)

    return (binary_mask * 255).astype(np.uint8)


def render_mask_overlay(frame_rgb: np.ndarray, mask_gray: np.ndarray,
                        alpha: float = 0.5, logger=None) -> np.ndarray:
    """Render a mask overlay on a frame."""
    logger = logger or EnhancedLogger()

    if mask_gray is None:
        return frame_rgb

    h, w = frame_rgb.shape[:2]
    if mask_gray.shape[:2] != (h, w):
        mask_gray = cv2.resize(mask_gray, (w, h),
                               interpolation=cv2.INTER_NEAREST)
    m = (mask_gray > 128)

    red_layer = np.zeros_like(frame_rgb, dtype=np.uint8)
    red_layer[..., 0] = 255  # Red channel for RGB

    blended = cv2.addWeighted(frame_rgb, 1.0 - alpha, red_layer, alpha, 0.0)
    if m.ndim == 2:
        m = m[..., np.newaxis]
    elif m.ndim == 3 and m.shape[2] != 1:
        logger.warning("Unexpected mask shape. Skipping overlay.",
                       extra={'shape': m.shape})
        return frame_rgb
    out = np.where(m, blended, frame_rgb)
    return out


def rgb_to_pil(image_rgb: np.ndarray) -> Image.Image:
    """Convert RGB numpy array to PIL Image."""
    return Image.fromarray(image_rgb)


def create_frame_map(output_dir: Path, logger=None):
    """Load or create map from original frame number to sequential filename."""
    logger = logger or EnhancedLogger()

    logger.info("Loading frame map...", component="frames")
    frame_map_path = output_dir / "frame_map.json"

    if not frame_map_path.exists():
        thumb_files = sorted(
            list((output_dir / "thumbs").glob("frame_*.webp")),
            key=lambda p: int(re.search(r'frame_(\d+)', p.name).group(1))
        )
        return {
            int(re.search(r'frame_(\d+)', f.name).group(1)): f.name
            for f in thumb_files
        }

    try:
        with open(frame_map_path, 'r', encoding='utf-8') as f:
            frame_map_list = json.load(f)
        return {
            orig_num: f"frame_{i+1:06d}.webp"
            for i, orig_num in enumerate(sorted(frame_map_list))
        }
    except Exception:
        logger.error("Failed to parse frame_map.json. Using fallback.",
                     exc_info=True)
        return {}