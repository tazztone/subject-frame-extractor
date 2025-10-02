"""Person detection utilities using YOLO."""

import torch
from functools import lru_cache
from ultralytics import YOLO


class PersonDetector:
    def __init__(self, model="yolo11x.pt", imgsz=640, conf=0.3, device='cuda'):
        from app.core.config import Config
        from app.core.logging import UnifiedLogger
        from app.ml.downloads import download_model

        config = Config()
        logger = UnifiedLogger()

        if YOLO is None:
            raise ImportError("Ultralytics YOLO not installed.")

        model_path = config.DIRS['models'] / model
        model_path.parent.mkdir(exist_ok=True)

        model_url = (f"https://huggingface.co/Ultralytics/YOLO11/"
                     f"resolve/main/{model}")
        download_model(model_url, model_path, "YOLO person detector")

        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(str(model_path))
        self.model.to(self.device)
        self.imgsz = imgsz
        self.conf = conf
        logger.info("YOLO person detector loaded",
                    extra={'device': self.device, 'model': model})

    def detect_boxes(self, img_rgb):
        """Detect person bounding boxes in an RGB image."""
        res = self.model.predict(img_rgb, imgsz=self.imgsz, conf=self.conf,
                                 classes=[0], verbose=False,
                                 device=self.device)
        boxes = []
        for r in res:
            if getattr(r, "boxes", None) is None:
                continue
            cpu_boxes = r.boxes.cpu()
            for b in cpu_boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                score = float(b.conf[0])
                boxes.append((x1, y1, x2, y2, score))
        return boxes


@lru_cache(maxsize=None)
def get_person_detector(model_name, device):
    """Load and cache a person detector model."""
    from app.core.logging import UnifiedLogger
    logger = UnifiedLogger()

    logger.info(f"Loading or getting cached person detector: {model_name}")
    return PersonDetector(model=model_name, device=device)
