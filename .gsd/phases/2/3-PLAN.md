---
phase: 2
plan: 3
wave: 2
---

# Plan 2.3: Face Metrics & Context Integration

## Objective
Migrate `eyes_open`, `yaw`, `pitch` metrics. These require face landmarks. We will implement them as light operators that consume detection results explicitly passed via `ctx.params`.

## Context
- @core/models.py — Current implementation (lines 150-177)
- @core/pipelines.py — Where detection happens (needs to be exposed)

## Tasks

<task type="auto">
  <name>Implement Face Metrics Operators</name>
  <files>
    - core/operators/face_metrics.py (NEW)
  </files>
  <action>
    Create `core/operators/face_metrics.py`:
    
    1. `FaceBlinkOperator`:
       - Config: name="eyes_open", category="face", requires_face=True
       - Execute: Read `ctx.params["face_landmarks"]`. Calculate eyes open score.
    
    2. `FacePoseOperator`:
       - Config: name="face_pose", category="face", requires_face=True
       - Execute: Read `ctx.params["face_matrix"]` (transformation matrix). Calculate `yaw`, `pitch`, `roll`.
       - Returns keys: `yaw`, `pitch`, `roll`.
    
    3. Ensure operators return `warning` if required params are missing from context.
  </action>
  <verify>python -c "from core.operators.face_metrics import FaceBlinkOperator; print('Face ops OK')"</verify>
  <done>Face operators implemented</done>
</task>

<task type="auto">
  <name>Test Face Metrics with Mock Data</name>
  <files>
    - tests/unit/test_face_operators.py (NEW)
  </files>
  <action>
    Create tests injecting mock landmark/matrix data into `ctx.params`:
    
    1. Test `FaceBlinkOperator` with open/closed eye blendshapes.
    2. Test `FacePoseOperator` with identity matrix (0 yaw/pitch) and rotated matrix.
    3. Verify missing params trigger warnings, not crashes.
  </action>
  <verify>uv run pytest tests/unit/test_face_operators.py -v</verify>
  <done>Tests pass with mock context params</done>
</task>

## Success Criteria
- [ ] Face operators consume standard data from `ctx.params`
- [ ] Logic accurately reflects `core/models.py` math
- [ ] Robust handling of missing face data
