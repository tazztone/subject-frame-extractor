"""
Face clustering and representative selection logic.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

if TYPE_CHECKING:
    from core.context import AnalysisContext
    from core.events import PreAnalysisEvent


def cluster_faces(all_faces: List[Dict[str, Any]], confidence: float = 0.5) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Groups detected faces into clusters based on embedding similarity.

    Args:
        all_faces: List of face detection dictionaries with 'embedding' keys.
        confidence: Clustering sensitivity (0.0 to 1.0).

    Returns:
        Tuple of (labels array, gallery_to_cluster_map).
    """
    if not all_faces:
        return np.array([]), {}

    embeddings = np.array([face["embedding"] for face in all_faces])

    # eps is distance, so higher confidence means smaller distance (tighter clusters)
    eps = 1.0 - confidence
    clustering = DBSCAN(eps=eps, min_samples=2, metric="cosine").fit(embeddings)

    # Create a mapping from gallery index to cluster label
    unique_labels = sorted([l for l in set(clustering.labels_) if l != -1])
    gallery_to_cluster_map = {i: label for i, label in enumerate(unique_labels)}

    return clustering.labels_, gallery_to_cluster_map


def get_cluster_representative(
    all_faces: List[Dict[str, Any]], labels: np.ndarray, target_label: int, video_path: str, output_dir: str
) -> Tuple[Optional[str], Optional[np.ndarray], str]:
    """
    Finds the best quality face in a cluster and saves a reference crop.

    Returns:
        Tuple of (crop_path, crop_image, status_message)
    """

    cluster_faces_list = [all_faces[i] for i, l in enumerate(labels) if l == target_label]
    if not cluster_faces_list:
        return None, None, "⚠️ Cluster not found"

    # Select face with highest detection score as the representative
    best_face = max(cluster_faces_list, key=lambda x: x["det_score"])

    # Extract high-res crop from original video
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, best_face["frame_num"])
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, None, "⚠️ Could not read video frame"

    # Map bbox from thumbnail coordinates to full frame
    # We need the thumbnail shape to compute the scale
    # If not provided, we might have to infer it or rely on consistent scaling
    # For now, we assume the bbox is already in 0-1 normalized coordinates or we need the thumb
    # AppUI logic used: x1, y1, x2, y2 = int(x1 * fw / w), int(y1 * fh / h)...
    # This implies best_face["bbox"] was in thumb-space.

    # To be fully decoupled, we'd prefer normalized coordinates,
    # but let's stick to the current logic for a surgical refactor.

    # [Refactoring Note: AppUI had access to thumbnail_manager.get(Path(best_face["thumb_path"]))]
    # We should probably pass the thumb dimensions or handle it here.

    thumb = cv2.imread(best_face["thumb_path"])
    if thumb is None:
        return None, None, "⚠️ Could not read thumbnail for scaling"

    h, w, _ = thumb.shape
    fh, fw, _ = frame.shape

    x1, y1, x2, y2 = best_face["bbox"]
    x1, y1, x2, y2 = int(x1 * fw / w), int(y1 * fh / h), int(x2 * fw / w), int(y2 * fh / h)

    # Ensure within bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)

    face_crop = frame[y1:y2, x1:x2]
    face_crop_path = Path(output_dir) / "reference_face.png"
    cv2.imwrite(str(face_crop_path), face_crop)

    # Return RGB for Gradio
    face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    return str(face_crop_path), face_crop_rgb, f"✅ **Selected Person {target_label}**"


def scan_faces_in_session(context: "AnalysisContext", params: "PreAnalysisEvent") -> List[Dict[str, Any]]:
    """Scans the video frames for faces using the face analyzer.

    Returns:
        List of face dictionaries containing:
        - frame_num: Frame number
        - bbox: Bounding box coordinates
        - embedding: Normed face embedding
        - det_score: Detection score
        - thumb_path: Path to the thumbnail image
    """
    from core.io_utils import create_frame_map
    from core.managers import initialize_analysis_models

    output_dir = Path(params.output_folder)
    context.logger.info(f"Output dir for face scanning: {output_dir}")

    # Initialize model
    models = initialize_analysis_models(params.model_dump(), context.config, context.logger, context.model_registry)
    face_analyzer = models.get("face_analyzer")
    if not face_analyzer:
        context.logger.warning("Face analyzer not available")
        return []

    frame_map = create_frame_map(output_dir, context.logger)
    context.logger.info(f"Frame map has {len(frame_map)} frames")
    if not frame_map:
        return []

    all_faces = []
    thumb_dir = output_dir / "thumbs"
    for frame_num, thumb_filename in frame_map.items():
        if context.cancel_event.is_set():
            context.logger.info("Face scanning cancelled")
            break
        if frame_num % params.pre_sample_nth != 0:
            continue
        thumb_rgb = context.thumbnail_manager.get(thumb_dir / thumb_filename)
        if thumb_rgb is None:
            continue
        faces = face_analyzer.get(cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2BGR))
        for face in faces:
            all_faces.append(
                {
                    "frame_num": frame_num,
                    "bbox": face.bbox,
                    "embedding": face.normed_embedding,
                    "det_score": face.det_score,
                    "thumb_path": str(thumb_dir / thumb_filename),
                }
            )
    return all_faces
