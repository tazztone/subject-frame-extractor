"""
Image Processing Utilities for Subject Frame Extractor
"""

from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from numba import njit
from PIL import Image

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger


def postprocess_mask(
    mask: np.ndarray, config: "Config", fill_holes: bool = True, keep_largest_only: bool = True
) -> np.ndarray:
    """Cleans up binary masks using morphological operations and connected components."""

    if mask is None or mask.size == 0:
        return mask
    binary_mask = (mask > 128).astype(np.uint8)
    if fill_holes:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (config.masking_close_kernel_size, config.masking_close_kernel_size)
        )
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    if keep_largest_only and config.masking_keep_largest_only:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            binary_mask = (labels == largest_label).astype(np.uint8)
    return (binary_mask * 255).astype(np.uint8)


def render_mask_overlay(frame_rgb: np.ndarray, mask_gray: np.ndarray, alpha: float, logger: "AppLogger") -> np.ndarray:
    """overlays a semi-transparent red mask on the image."""
    if mask_gray is None or frame_rgb is None:
        return frame_rgb if frame_rgb is not None else np.array([])
    h, w = frame_rgb.shape[:2]
    if mask_gray.shape[:2] != (h, w):
        mask_gray = cv2.resize(mask_gray, (w, h), interpolation=cv2.INTER_NEAREST)
    m = mask_gray > 128
    red_layer = np.zeros_like(frame_rgb, dtype=np.uint8)
    red_layer[..., 0] = 255
    blended = cv2.addWeighted(frame_rgb, 1.0 - alpha, red_layer, alpha, 0.0)
    if m.ndim == 2:
        m = m[..., np.newaxis]
    elif m.ndim == 3 and m.shape[2] != 1:
        logger.warning("Unexpected mask shape. Skipping overlay.", extra={"shape": m.shape})
        return frame_rgb
    return np.where(m, blended, frame_rgb)


def rgb_to_pil(image_rgb: np.ndarray) -> Image.Image:
    """Converts a NumPy RGB array to a PIL Image."""
    return Image.fromarray(image_rgb)


def draw_bbox(
    img_rgb: np.ndarray,
    xywh: list,
    config: "Config",
    color: Optional[tuple] = None,
    thickness: Optional[int] = None,
    label: Optional[str] = None,
) -> np.ndarray:
    """Draws a bounding box and optional label on an image."""
    color = color or tuple(config.visualization_bbox_color)
    thickness = thickness or config.visualization_bbox_thickness
    x, y, w, h = map(int, xywh or [0, 0, 0, 0])
    img_out = img_rgb.copy()
    cv2.rectangle(img_out, (x, y), (x + w, y + h), color, thickness)
    if label:
        font_scale = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = x + 5
        text_y = y + text_height + 5
        cv2.rectangle(img_out, (x, y), (x + text_width + 10, y + text_height + 10), color, -1)
        cv2.putText(img_out, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    return img_out


@njit
def compute_entropy(hist: np.ndarray, entropy_norm: float) -> float:
    """Computes normalized entropy from a histogram using Numba."""
    prob = hist / (np.sum(hist) + 1e-7)
    entropy = -np.sum(prob[prob > 0] * np.log2(prob[prob > 0]))
    return min(max(entropy / entropy_norm, 0), 1.0)
