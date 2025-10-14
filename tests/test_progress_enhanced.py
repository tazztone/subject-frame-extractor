import pytest
import time
from queue import Queue
from app.progress_enhanced import AdvancedProgressTracker, ProgressState

@pytest.fixture
def progress_tracker():
    """Fixture for AdvancedProgressTracker."""
    queue = Queue()
    return AdvancedProgressTracker(progress_queue=queue)

def test_start_operation(progress_tracker):
    """Test starting a new operation."""
    progress_tracker.start_operation("test_op", 100)
    assert progress_tracker.current_state is not None
    assert progress_tracker.current_state.operation == "test_op"
    assert progress_tracker.current_state.total == 100
    assert not progress_tracker.progress_queue.empty()

def test_start_stage(progress_tracker):
    """Test starting a new stage."""
    progress_tracker.start_operation("test_op", 100)
    progress_tracker.start_stage("stage1", 50)
    assert progress_tracker.current_state.stage == "stage1"
    assert progress_tracker.current_state.stage_total == 50
    assert not progress_tracker.progress_queue.empty()

def test_update_progress(progress_tracker):
    """Test updating progress."""
    progress_tracker.start_operation("test_op", 100)
    progress_tracker.update_progress(10)
    assert progress_tracker.current_state.current == 10
    assert not progress_tracker.progress_queue.empty()

def test_complete_operation(progress_tracker):
    """Test completing an operation."""
    progress_tracker.start_operation("test_op", 100)
    progress_tracker.complete_operation(success=True)
    assert progress_tracker.current_state.stage == "completed"
    assert len(progress_tracker.history) == 1
    assert not progress_tracker.progress_queue.empty()

if __name__ == "__main__":
    pytest.main([__file__])