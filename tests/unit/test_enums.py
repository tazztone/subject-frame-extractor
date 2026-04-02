from core.enums import PropagationDirection, SceneStatus, get_coco_id


def test_scene_status_values():
    """Verify SceneStatus enum values."""
    assert SceneStatus.PENDING == "pending"
    assert SceneStatus.INCLUDED == "included"
    assert SceneStatus.EXCLUDED == "excluded"
    assert str(SceneStatus.PENDING) == "pending"


def test_propagation_direction_values():
    """Verify PropagationDirection enum values."""
    assert PropagationDirection.FORWARD == "forward"
    assert PropagationDirection.BACKWARD == "backward"
    assert PropagationDirection.BOTH == "both"
    assert str(PropagationDirection.FORWARD) == "forward"


def test_get_coco_id_basic():
    """Verify standard class name resolution."""
    assert get_coco_id("person") == 0
    assert get_coco_id("dog") == 16
    assert get_coco_id("bicycle") == 1


def test_get_coco_id_case_insensitive():
    """Verify case-insensitivity in resolution."""
    assert get_coco_id("Person") == 0
    assert get_coco_id("DOG") == 16
    assert get_coco_id("BiCyCLe") == 1


def test_get_coco_id_fallback():
    """Verify fallback to person for unknown or empty input."""
    assert get_coco_id("nonexistent_class") == 0
    assert get_coco_id("") == 0
    assert get_coco_id(None) == 0
