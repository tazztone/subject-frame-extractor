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
    FIND_PROMINENT = "Find Prominent Person"

    def __str__(self) -> str:
        return self.value
