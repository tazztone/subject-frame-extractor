"""
Centralized model management for initializing and loading analysis models.
This ensures that model loading logic is not duplicated across different parts
of the application, such as pipeline_logic.py and scene_logic.py.
"""

from pathlib import Path
import cv2
import numpy as np

from app.config import Config
from app.logging_enhanced import EnhancedLogger
from app.models import AnalysisParameters
from app.face import get_face_analyzer
from app.person import get_person_detector


def initialize_analysis_models(params: AnalysisParameters, config: Config,
                               logger: EnhancedLogger, cuda_available: bool):
    """
    Initializes and returns all models needed for analysis based on params.
    This includes the face analyzer, reference face embedding, and person detector.
    """
    device = "cuda" if cuda_available else "cpu"
    face_analyzer, ref_emb, person_detector = None, None, None

    # Initialize Face Analyzer and Reference Embedding if needed
    if params.enable_face_filter:
        face_analyzer = get_face_analyzer(params.face_model_name, config, logger)
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
                        else:
                            logger.warning("No face found in reference image.", extra={'path': ref_path})
                    else:
                        logger.warning("Could not read reference face image.", extra={'path': ref_path})
                except Exception as e:
                    logger.error("Failed to process reference face image.", exc_info=True)
            else:
                logger.warning("Reference face image path does not exist.", extra={'path': ref_path})

    # Initialize Person Detector
    person_detector = get_person_detector(params.person_detector_model, device, config, logger)

    return {
        "face_analyzer": face_analyzer,
        "ref_emb": ref_emb,
        "person_detector": person_detector,
        "device": device
    }