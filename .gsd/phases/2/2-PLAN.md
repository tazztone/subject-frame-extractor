---
phase: 2
plan: 2
wave: 1
---

# Plan 2.2: NIQE Operator Migration

## Objective
Migrate the `niqe` (Natural Image Quality Evaluator) metric. This requires stateful initialization to load the PyTorch model (`pyiqa`) and handling GPU/CPU devices.

## Context
- @core/models.py — Current implementation (lines 236-258)
- @core/operators/base.py — Operator Protocol (initialize lifecycle)

## Tasks

<task type="auto">
  <name>Implement NIQE Operator</name>
  <files>
    - core/operators/niqe.py (NEW)
  </files>
  <action>
    Create `core/operators/niqe.py`:
    
    1. `NiqeOperator` class:
       - Config: name="niqe", category="quality", min=0, max=100 (it's a score, typically lower is better, but check normalization)
       - `initialize(self, config)`:
         - Import `pyiqa` inside method (lazy import).
         - Load model to `self.model`.
         - Handle device selection (cuda vs cpu).
       - `execute(self, ctx)`:
         - Check if initialized.
         - Preprocess image: `ctx.image_rgb` -> tensor (HWC -> CHW, /255.0).
         - Send to device.
         - Run inference.
         - Normalize/Scale if needed (current implementation doesn't seem to scale explicitly in snippet, check logic).
         - Return `metrics={"niqe_score": val}`.
    
    2. Handle errors gracefully (e.g., pyiqa not installed).
  </action>
  <verify>python -c "from core.operators.niqe import NiqeOperator; print(NiqeOperator().config.name)"</verify>
  <done>NIQE operator implemented with lazy loading</done>
</task>

<task type="auto">
  <name>Add NIQE Unit/Mock Tests</name>
  <files>
    - tests/unit/test_niqe_operator.py (NEW)
  </files>
  <action>
    Create tests mocking `pyiqa`:
    
    1. Test `initialize`: calls `pyiqa.create_metric`.
    2. Test `execute`: transforms input, calls model, returns score.
    3. Test error handling: returns error in OperatorResult if model fails.
    
    Do NOT require GPU or actual model loading for unit tests.
  </action>
  <verify>uv run pytest tests/unit/test_niqe_operator.py -v</verify>
  <done>Tests pass with mocked model</done>
</task>

## Success Criteria
- [ ] `NiqeOperator` implements `initialize` for model loading
- [ ] Graceful fallback if `pyiqa` missing
- [ ] Tests verify lifecycle and execution flow
