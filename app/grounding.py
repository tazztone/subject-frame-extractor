"""Grounding DINO model utilities."""

from pathlib import Path
import torch
from functools import lru_cache
from grounding_dino.groundingdino.util.inference import (
    load_model as gdino_load_model,
    predict as gdino_predict,
)


@lru_cache(maxsize=None)
def get_grounding_dino_model(gdino_config_path: str, gdino_checkpoint_path: str,
                           config: 'Config', device="cuda", logger=None):
    """Load and cache the Grounding DINO model."""
    from app.logging_enhanced import EnhancedLogger
    from app.downloads import download_model
    from app.error_handling import ErrorHandler

    logger = logger or EnhancedLogger()
    error_handler = ErrorHandler(logger, config)

    try:
        ckpt_path = Path(gdino_checkpoint_path)
        if not ckpt_path.is_absolute():
            ckpt_path = config.DIRS['models'] / ckpt_path.name

        download_model(
            "https://github.com/IDEA-Research/GroundingDINO/releases/"
            "download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
            ckpt_path,
            "GroundingDINO Swin-T model",
            logger,
            error_handler,
            min_size=500_000_000
        )

        gdino_model = gdino_load_model(
            model_config_path=gdino_config_path,
            model_checkpoint_path=str(ckpt_path),
            device=device,
        )
        logger.info("Grounding DINO model loaded.",
                   component="grounding",
                   user_context={'model_path': str(ckpt_path)})
        return gdino_model
    except Exception as e:
        logger.warning("Grounding DINO unavailable.", component="grounding", exc_info=True)
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
