"""Configuration management for the frame extractor application."""

import yaml
from pathlib import Path


class Config:
    BASE_DIR = Path(__file__).parent.parent.parent
    DIRS = {
        'logs': BASE_DIR / "logs",
        'configs': BASE_DIR / "configs",
        'models': BASE_DIR / "models",
        'downloads': BASE_DIR / "downloads"
    }
    CONFIG_FILE = DIRS['configs'] / "config.yaml"

    def __init__(self):
        self.settings = self.load_config()
        for key, value in self.settings.items():
            setattr(self, key, value)

        self.thumbnail_cache_size = self.settings.get(
            'thumbnail_cache_size', 200
        )

        self.GROUNDING_DINO_CONFIG = (
            self.BASE_DIR / self.model_paths['grounding_dino_config']
        )
        self.GROUNDING_DINO_CKPT = (
            self.DIRS['models'] /
            Path(self.model_paths['grounding_dino_checkpoint']).name
        )
        self.GROUNDING_BOX_THRESHOLD = (
            self.grounding_dino_params['box_threshold']
        )
        self.GROUNDING_TEXT_THRESHOLD = (
            self.grounding_dino_params['text_threshold']
        )
        self.QUALITY_METRICS = list(self.quality_weights.keys())

    def load_config(self):
        self.DIRS['configs'].mkdir(exist_ok=True)
        if not self.CONFIG_FILE.exists():
            raise FileNotFoundError(
                f"Configuration file not found at {self.CONFIG_FILE}. "
                "Please ensure it exists."
            )
        with open(self.CONFIG_FILE, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @classmethod
    def setup_directories_and_logger(cls):
        from app.core.logging import UnifiedLogger
        for dir_path in cls.DIRS.values():
            dir_path.mkdir(exist_ok=True)
        return UnifiedLogger()
