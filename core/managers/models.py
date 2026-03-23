from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import lpips
import torch

from core.error_handling import ErrorHandler
from core.io_utils import download_model

from .face import get_face_analyzer, get_face_landmarker

if TYPE_CHECKING:
    from core.config import Config
    from core.logger import AppLogger
    from core.models import AnalysisParameters

    from .registry import ModelRegistry


def get_lpips_metric(model_name: str = "alex", device: str = "cpu"):
    """Returns the LPIPS metric model."""
    return lpips.LPIPS(net=model_name).to(device)


def initialize_analysis_models(
    params: "AnalysisParameters", config: "Config", logger: "AppLogger", model_registry: "ModelRegistry"
) -> dict:
    """Initializes all necessary analysis models based on parameters."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    face_analyzer, ref_emb, face_landmarker = None, None, None

    if params.enable_face_filter:
        face_analyzer = get_face_analyzer(
            model_name=params.face_model_name,
            models_path=str(config.models_dir),
            det_size_tuple=tuple(config.model_face_analyzer_det_size),
            logger=logger,
            model_registry=model_registry,
            device=device,
        )
        if face_analyzer and params.face_ref_img_path:
            ref_path = Path(params.face_ref_img_path)
            if ref_path.exists() and ref_path.is_file():
                try:
                    ref_img = cv2.imread(str(ref_path))
                    if ref_img is not None:
                        faces = face_analyzer.get(ref_img)
                        if faces:
                            ref_emb = max(faces, key=lambda x: x.det_score).normed_embedding
                            logger.info("Reference face embedding created successfully.")
                except Exception as e:
                    logger.error(f"Failed to process reference face: {e}")

    landmarker_path = Path(config.models_dir) / Path(config.face_landmarker_url).name
    download_model(
        config.face_landmarker_url,
        landmarker_path,
        "MediaPipe Face Landmarker",
        logger,
        ErrorHandler(logger, config.retry_max_attempts, config.retry_backoff_seconds),
        config.user_agent,
        expected_sha256=config.face_landmarker_sha256,
    )
    if landmarker_path.exists():
        face_landmarker = get_face_landmarker(str(landmarker_path), logger)

    return {"face_analyzer": face_analyzer, "ref_emb": ref_emb, "face_landmarker": face_landmarker, "device": device}
