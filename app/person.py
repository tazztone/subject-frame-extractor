"""Person detection utilities using YOLO."""

import torch
from functools import lru_cache
from ultralytics import YOLO


class PersonDetector:
    def __init__(self, model="yolo11x.pt", imgsz=640, conf=0.3, device='cuda', config=None, logger=None):
        from app.logging_enhanced import EnhancedLogger
        from app.downloads import download_model
        from app.error_handling import ErrorHandler
        from app.config import Config

        self.config = config or Config()
        self.logger = logger or EnhancedLogger()
        error_handler = ErrorHandler(self.logger, self.config)

        if YOLO is None:
            raise ImportError("Ultralytics YOLO not installed.")

        model_path = config.DIRS['models'] / model
        model_path.parent.mkdir(exist_ok=True)

        model_url = (f"https://huggingface.co/Ultralytics/YOLO11/"
                     f"resolve/main/{model}")
        download_model(model_url, model_path, "YOLO person detector", self.logger, error_handler)

        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(str(model_path))
        self.model.to(self.device)
        self.imgsz = imgsz
        self.conf = conf
        self.logger.info("YOLO person detector loaded",
                         component="person_detector",
                         user_context={'device': self.device, 'model': model})

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
def get_person_detector(model_name, device, config: 'Config', logger=None):
    """Load and cache a person detector model."""
    from app.logging_enhanced import EnhancedLogger
    logger = logger or EnhancedLogger()

    logger.info(f"Loading or getting cached person detector: {model_name}", component="person_detector")
    return PersonDetector(model=model_name, device=device, config=config, logger=logger)
