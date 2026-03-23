"""
Advanced cropping operators and geometry calculations.
"""

from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def calculate_best_crop(
    frame_w: int,
    frame_h: int,
    subject_bbox: tuple[int, int, int, int],
    aspect_ratios: List[tuple[str, float]],
    padding_factor: float = 1.0,
) -> Optional[Dict[str, Any]]:
    """Find the best-fitting crop region for a subject based on desired aspect ratios."""
    x_b, y_b, w_b, h_b = subject_bbox
    if w_b == 0 or h_b == 0:
        return None

    feasible_candidates = []
    for ar_str, r in aspect_ratios:
        if w_b / h_b > r:
            w_c, h_c = w_b, w_b / r
        else:
            h_c, w_c = h_b, h_b * r

        w_padded, h_padded = w_c * padding_factor, h_c * padding_factor

        scale = 1.0
        if w_padded > frame_w:
            scale = min(scale, frame_w / w_padded)
        if h_padded > frame_h:
            scale = min(scale, frame_h / h_padded)

        w_final, h_final = w_padded * scale, h_padded * scale

        if w_final < w_b or h_final < h_b:
            if w_final < w_b:
                w_final = w_b
                h_final = w_final / r
            if h_final < h_b:
                h_final = h_b
                w_final = h_final * r
            if w_final > frame_w:
                w_final = frame_w
                h_final = w_final / r
            if h_final > frame_h:
                h_final = frame_h
                w_final = h_final * r

        center_x_b, center_y_b = x_b + w_b / 2, y_b + h_b / 2
        x1 = center_x_b - w_final / 2
        y1 = center_y_b - h_final / 2

        # Clamp to frame boundaries
        x1 = max(0, min(x1, frame_w - w_final))
        y1 = max(0, min(y1, frame_h - h_final))

        # Only accept if it still contains the subject
        if x1 > x_b or y1 > y_b or x1 + w_final < x_b + w_b or y1 + h_final < y_b + h_b:
            continue

        feasible_candidates.append(
            {
                "ar_str": ar_str,
                "x": int(x1),
                "y": int(y1),
                "w": int(w_final),
                "h": int(h_final),
                "area": w_final * h_final,
            }
        )

    if not feasible_candidates:
        return None

    # Choose smallest area crop that matches aspect ratio best
    subject_ar = w_b / h_b
    return min(feasible_candidates, key=lambda c: (c["area"], abs((c["w"] / c["h"]) - subject_ar)))


def crop_image_with_subject(
    image: np.ndarray,
    mask: np.ndarray,
    aspect_ratios: List[tuple[str, float]],
    padding_factor: float = 1.0,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Crop an image to focus on the subject defined by a mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, "No subject found in mask."

    bbox = cv2.boundingRect(np.concatenate(contours))
    h, w = image.shape[:2]

    best_crop = calculate_best_crop(w, h, bbox, aspect_ratios, padding_factor)

    if best_crop:
        x, y, cw, ch = best_crop["x"], best_crop["y"], best_crop["w"], best_crop["h"]
        return image[y : y + ch, x : x + cw], best_crop["ar_str"]
    else:
        # Fallback to tight bbox if no aspect ratio fits perfectly
        x, y, bw, bh = bbox
        return image[y : y + bh, x : x + bw], "native"
