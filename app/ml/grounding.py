"""Grounding DINO model utilities."""

from pathlib import Path
import torch
from grounding_dino.groundingdino.util.inference import (
    load_model as gdino_load_model,
    predict as gdino_predict,
)


def load_grounding_dino_model(params, device="cuda"):
    """Load and initialize Grounding DINO model."""
    from app.core.config import Config
    from app.core.logging import UnifiedLogger
    from app.ml.downloads import download_model
    
    config = Config()
    logger = UnifiedLogger()
    
    try:
        ckpt_path = Path(params.gdino_checkpoint_path)
        if not ckpt_path.is_absolute():
            ckpt_path = config.DIRS['models'] / ckpt_path.name
            
        download_model(
            "https://github.com/IDEA-Research/GroundingDINO/releases/"
            "download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
            ckpt_path, "GroundingDINO Swin-T model", min_size=500_000_000
        )
        
        gdino_model = gdino_load_model(
            model_config_path=params.gdino_config_path,
            model_checkpoint_path=str(ckpt_path),
            device=device,
        )
        logger.info("Grounding DINO model loaded.",
                   extra={'model_path': str(ckpt_path)})
        return gdino_model
    except Exception as e:
        logger.warning("Grounding DINO unavailable.", exc_info=True)
        return None


def predict_grounding_dino(model, image_tensor, caption, box_threshold,
                          text_threshold, device="cuda"):
    """Run Grounding DINO prediction."""
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == 'cuda')):
        return gdino_predict(
            model=model,
            image=image_tensor.to(device),
            caption=caption,
            box_threshold=float(box_threshold),
            text_threshold=float(text_threshold),
        )
