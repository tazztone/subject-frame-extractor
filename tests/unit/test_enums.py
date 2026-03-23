from core.enums import PropagationDirection, SceneStatus


def test_scene_status():
    """Test SceneStatus enum."""
    assert SceneStatus.PENDING == "pending"
    assert str(SceneStatus.INCLUDED) == "included"
    assert SceneStatus("excluded") == SceneStatus.EXCLUDED


def test_propagation_direction():
    """Test PropagationDirection enum."""
    assert PropagationDirection.FORWARD == "forward"
    assert str(PropagationDirection.BOTH) == "both"
    assert PropagationDirection("backward") == PropagationDirection.BACKWARD
