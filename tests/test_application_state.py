
import pytest
from ui.app_ui import ApplicationState

def test_push_pop_history():
    state = ApplicationState()
    
    # Create some dummy scenes
    scene1 = [{"id": 1, "start": 0, "end": 10}]
    scene2 = [{"id": 1, "start": 0, "end": 10}, {"id": 2, "start": 20, "end": 30}]
    
    # Test push
    state.push_history(scene1)
    assert len(state.scene_history) == 1
    assert state.scene_history[0] == scene1
    
    # Test deep copy
    scene1[0]["start"] = 5
    assert state.scene_history[0][0]["start"] == 0 # Should remain 0
    
    state.push_history(scene2)
    assert len(state.scene_history) == 2
    assert state.scene_history[1] == scene2
    
    # Test pop
    popped = state.pop_history()
    assert popped == scene2
    assert len(state.scene_history) == 1
    
    popped = state.pop_history()
    assert popped[0]["start"] == 0 # Should be the original version
    assert len(state.scene_history) == 0
    
    # Test empty pop
    assert state.pop_history() is None

def test_history_max_depth():
    state = ApplicationState()
    for i in range(15):
        state.push_history([{"id": i}])
        
    assert len(state.scene_history) == 10
    assert state.scene_history[0][0]["id"] == 5 # First 5 should involve been dropped
    assert state.scene_history[-1][0]["id"] == 14
