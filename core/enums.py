from enum import Enum


class SceneStatus(str, Enum):
    """Status of a scene/shot for inclusion in the final dataset."""

    PENDING = "pending"
    INCLUDED = "included"
    EXCLUDED = "excluded"

    def __str__(self) -> str:
        return self.value


class PropagationDirection(str, Enum):
    """Direction for SAM3 mask propagation."""

    FORWARD = "forward"
    BACKWARD = "backward"
    BOTH = "both"

    def __str__(self) -> str:
        return self.value


class SeedStrategy(str, Enum):
    """Strategies for initial subject seeding."""

    FACE_REFERENCE = "Source Face Reference"
    TEXT_DESCRIPTION = "Text Description (Limited)"
    FACE_TEXT_FALLBACK = "Face + Text Fallback"
    AUTOMATIC = "Automatic Detection"

    def __str__(self) -> str:
        return self.value


COCO_CLASSES: list[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

COCO_CLASS_ID: dict[str, int] = {name: i for i, name in enumerate(COCO_CLASSES)}


def get_coco_id(name: str) -> int:
    """Safely resolves a class name to its COCO ID."""
    if not name:
        return 0
    return COCO_CLASS_ID.get(name.lower(), 0)


ANCHOR_STRATEGIES: list[str] = [
    "Largest Subject",
    "Center-most Subject",
    "Highest Confidence",
    "Tallest Subject",
    "Area x Confidence",
    "Rule-of-Thirds",
    "Edge-avoiding",
    "Balanced",
    "Best Face",
]
