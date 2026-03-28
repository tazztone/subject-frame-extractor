from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

if TYPE_CHECKING:
    from insightface.app import FaceAnalysis

    from core.logger import AppLogger

    from .registry import ModelRegistry

thread_local = threading.local()


def get_face_landmarker(model_path: str, logger: "AppLogger") -> Any:
    """Returns a thread-local MediaPipe FaceLandmarker instance."""
    if hasattr(thread_local, "face_landmarker_instance"):
        return thread_local.face_landmarker_instance
    logger.info("Initializing MediaPipe FaceLandmarker for new thread.", component="face_landmarker")
    try:
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
        )
        detector = vision.FaceLandmarker.create_from_options(options)
        thread_local.face_landmarker_instance = detector
        logger.success("Face landmarker model initialized successfully for this thread.")
        return detector

    except Exception as e:
        logger.error(f"Could not initialize MediaPipe face landmarker model. Error: {e}", component="face_landmarker")
        raise RuntimeError("Could not initialize MediaPipe face landmarker model.") from e


def get_face_analyzer(
    model_name: str,
    models_path: str,
    det_size_tuple: tuple,
    logger: "AppLogger",
    model_registry: "ModelRegistry",
    device: str = "cpu",
) -> "FaceAnalysis":
    """Gets or loads the InsightFace FaceAnalysis app, with OOM handling."""
    from insightface.app import FaceAnalysis

    model_key = f"face_analyzer_{model_name}_{device}_{det_size_tuple}"

    def _loader():
        logger.info(f"Loading face model: {model_name} on device: {device}")
        try:
            is_cuda = device == "cuda"
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if is_cuda else ["CPUExecutionProvider"]
            analyzer = FaceAnalysis(name=model_name, root=models_path, providers=providers)
            analyzer.prepare(ctx_id=0 if is_cuda else -1, det_size=det_size_tuple)
            logger.success(f"Face model loaded with {'CUDA' if is_cuda else 'CPU'}.")
            return analyzer
        except Exception as e:
            import torch

            if "out of memory" in str(e) and device == "cuda":
                torch.cuda.empty_cache()
                logger.warning("CUDA OOM, retrying with CPU...")
                try:
                    analyzer = FaceAnalysis(name=model_name, root=models_path, providers=["CPUExecutionProvider"])
                    analyzer.prepare(ctx_id=-1, det_size=det_size_tuple)
                    return analyzer
                except Exception as cpu_e:
                    logger.error(f"CPU fallback also failed: {cpu_e}")
            raise RuntimeError(f"Could not initialize face analysis model. Error: {e}") from e

    return model_registry.get_or_load(model_key, _loader)
