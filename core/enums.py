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
