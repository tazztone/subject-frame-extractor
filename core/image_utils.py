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


def _draw_dashed_line(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash_len: int = 10,
):
    """Internal helper to draw a dashed line."""
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    if dist == 0:
        return
    for i in np.arange(0, dist, dash_len * 2):
        r1 = i / dist
        r2 = min(i + dash_len, dist) / dist
        p1 = (int(pt1[0] + (pt2[0] - pt1[0]) * r1), int(pt1[1] + (pt2[1] - pt1[1]) * r1))
        p2 = (int(pt1[0] + (pt2[0] - pt1[0]) * r2), int(pt1[1] + (pt2[1] - pt1[1]) * r2))
        cv2.line(img, p1, p2, color, thickness)


def _draw_dotted_line(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    gap: int = 5,
):
    """Internal helper to draw a dotted line."""
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    if dist == 0:
        return
    for i in np.arange(0, dist, gap * 2):
        r = i / dist
        p = (int(pt1[0] + (pt2[0] - pt1[0]) * r), int(pt1[1] + (pt2[1] - pt1[1]) * r))
        cv2.circle(img, p, thickness, color, -1)


def draw_bbox(
    img_rgb: np.ndarray,
    xywh: list,
    config: "Config",
    color: Optional[tuple] = None,
    thickness: Optional[int] = None,
    label: Optional[str] = None,
    style: Optional[str] = None,
    radius: Optional[int] = None,
    inplace: bool = False,
) -> np.ndarray:
    """Draws a bounding box and optional label on an image."""
    color = color or tuple(config.visualization_bbox_color)
    thickness = thickness or config.visualization_bbox_thickness
    style = style if style is not None else getattr(config, "visualization_bbox_style", "solid")
    radius = radius if radius is not None else getattr(config, "visualization_bbox_radius", 0)
    if not isinstance(radius, (int, float)):
        radius = 0
    show_labels = getattr(config, "visualization_show_labels", True)

    x, y, w, h = map(int, xywh or [0, 0, 0, 0])
    x2, y2 = x + w, y + h
    img_out = img_rgb if inplace else img_rgb.copy()

    if radius > 0:
        # Rounded corners
        r = min(int(radius), w // 2, h // 2)
        cv2.line(img_out, (x + r, y), (x2 - r, y), color, thickness)
        cv2.line(img_out, (x + r, y2), (x2 - r, y2), color, thickness)
        cv2.line(img_out, (x, y + r), (x, y2 - r), color, thickness)
        cv2.line(img_out, (x2, y + r), (x2, y2 - r), color, thickness)
        cv2.ellipse(img_out, (x + r, y + r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img_out, (x2 - r, y + r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img_out, (x + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        cv2.ellipse(img_out, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)
    elif style == "dashed":
        _draw_dashed_line(img_out, (x, y), (x2, y), color, thickness)
        _draw_dashed_line(img_out, (x, y2), (x2, y2), color, thickness)
        _draw_dashed_line(img_out, (x, y), (x, y2), color, thickness)
        _draw_dashed_line(img_out, (x2, y), (x2, y2), color, thickness)
    elif style == "dotted":
        _draw_dotted_line(img_out, (x, y), (x2, y), color, thickness)
        _draw_dotted_line(img_out, (x, y2), (x2, y2), color, thickness)
        _draw_dotted_line(img_out, (x, y), (x, y2), color, thickness)
        _draw_dotted_line(img_out, (x2, y), (x2, y2), color, thickness)
    else:
        cv2.rectangle(img_out, (x, y), (x2, y2), color, thickness)

    if label and (show_labels is True):
        font_scale = getattr(config, "visualization_label_font_scale", 0.5)
        if not isinstance(font_scale, (int, float)):
            font_scale = 0.5
        font_thickness = getattr(config, "visualization_label_thickness", 1)
        if not isinstance(font_thickness, int):
            font_thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        text_x = x
        text_y = y - 5 if y - 5 > text_height else y + text_height + 5
        cv2.rectangle(img_out, (text_x, text_y - text_height - 5), (text_x + text_width + 10, text_y + 5), color, -1)
        cv2.putText(img_out, label, (text_x + 5, text_y), font, font_scale, (255, 255, 255), font_thickness)
    return img_out


@njit
def compute_entropy(hist: np.ndarray, entropy_norm: float) -> float:
    """Computes normalized entropy from a histogram using Numba."""
    prob = hist / (np.sum(hist) + 1e-7)
    entropy = -np.sum(prob[prob > 0] * np.log2(prob[prob > 0]))
    return min(max(entropy / entropy_norm, 0), 1.0)
