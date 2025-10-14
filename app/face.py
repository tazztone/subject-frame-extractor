"""Face analysis utilities using InsightFace."""

import torch
from functools import lru_cache
from insightface.app import FaceAnalysis


@lru_cache(maxsize=None)
def get_face_analyzer(model_name, logger: 'EnhancedLogger'):
    """Load and cache a face analysis model."""
    from app.config import Config

    config = Config()

    logger.info(f"Loading or getting cached face model: {model_name}")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                     if device == 'cuda' else ['CPUExecutionProvider'])
        analyzer = FaceAnalysis(name=model_name,
                               root=str(config.DIRS['models']),
                               providers=providers)
        analyzer.prepare(ctx_id=0 if device == 'cuda' else -1,
                         det_size=(640, 640))
        device_str = 'CUDA' if device == 'cuda' else 'CPU'
        logger.success(f"Face model loaded with {device_str}.")
        return analyzer
    except Exception as e:
        raise RuntimeError(
            f"Could not initialize face analysis model. Error: {e}"
        ) from e
