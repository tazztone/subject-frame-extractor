---
phase: 2
plan: 4
wave: 3
---

# Plan 2.4: Face Metrics Migration

## Objective
Migrate `eyes_open`, `yaw`, `pitch` metrics. These require face detection data, which the pipeline must populate into `ctx.params` before calling operators.

## Context
- @core/models.py — Current implementation (lines 150-177)
- @core/pipelines.py — Face detection in `_analyze_face_similarity` and MediaPipe landmarker

## Tasks

<task type="auto">
  <name>Implement Face Metrics Operators</name>
  <files>
    - core/operators/face_metrics.py (NEW)
  </files>
  <action>
    Create `core/operators/face_metrics.py`:
    
    1. `EyesOpenOperator`:
       - Config: name="eyes_open", category="face", requires_face=True
       - Execute: 
         - Read `ctx.params.get("face_blendshapes")`.
         - If missing: return warning, score=None.
         - Calculate: `1.0 - max(eyeBlinkLeft, eyeBlinkRight)`.
         - Also calculate `blink_prob` for backward compat.
         - Return: `{"eyes_open_score": 0-100, "blink_prob": 0-1}`
    
    2. `FacePoseOperator`:
       - Config: name="face_pose", category="face", requires_face=True
       - Execute:
         - Read `ctx.params.get("face_matrix")` (4x4 transformation matrix).
         - If missing: return warning.
         - Calculate yaw, pitch, roll using atan2 math from models.py.
         - Return: `{"yaw": degrees, "pitch": degrees, "roll": degrees}`
    
    Handle missing data gracefully (warning, not error).
  </action>
  <verify>uv run python -c "from core.operators.face_metrics import EyesOpenOperator, FacePoseOperator; print('OK')"</verify>
  <done>Face operators implemented</done>
</task>

<task type="auto">
  <name>Update Pipeline to Populate Face Context</name>
  <files>
    - core/pipelines.py (MODIFY)
  </files>
  <action>
    Modify `AnalysisPipeline._process_single_frame`:
    
    1. Run face detection/landmarking BEFORE calling `run_operators`.
    2. Create context params dict:
       ```python
       face_params = {}
       if face_landmarker_result:
           if face_landmarker_result.face_blendshapes:
               face_params["face_blendshapes"] = {
                   b.category_name: b.score 
                   for b in face_landmarker_result.face_blendshapes[0]
               }
           if face_landmarker_result.facial_transformation_matrixes:
               face_params["face_matrix"] = face_landmarker_result.facial_transformation_matrixes[0]
       ```
    3. Pass to `run_operators(..., params=face_params)`.
  </action>
  <verify>uv run pytest tests/unit/test_face_operators.py -v</verify>
  <done>Pipeline populates face context; operators consume it</done>
</task>

<task type="auto">
  <name>Test Face Operators with Mock Data</name>
  <files>
    - tests/unit/test_face_operators.py (NEW)
  </files>
  <action>
    Create tests injecting mock data into `ctx.params`:
    
    1. `EyesOpenOperator`:
       - Open eyes (blink=0.0) → score=100
       - Closed eyes (blink=1.0) → score=0
       - Missing blendshapes → warning
    
    2. `FacePoseOperator`:
       - Identity matrix → yaw=0, pitch=0, roll=0
       - Rotated matrix → expected angles
       - Missing matrix → warning
  </action>
  <verify>uv run pytest tests/unit/test_face_operators.py -v</verify>
  <done>Tests pass with mock context</done>
</task>

## Success Criteria
- [ ] Face operators consume `ctx.params` populated by pipeline
- [ ] Math matches `core/models.py` exactly
- [ ] Robust handling of missing face data
