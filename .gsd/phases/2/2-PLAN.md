---
phase: 2
plan: 2
wave: 1
---

# Plan 2.2: NIQE Operator Migration

## Objective
Migrate the `niqe` (Natural Image Quality Evaluator) metric. This requires stateful initialization to load the PyTorch model (`pyiqa`) and handling GPU/CPU devices.

## Context
- @core/models.py — Current implementation (lines 236-280)
- @core/operators/base.py — Operator Protocol (initialize lifecycle)
- @core/pipelines.py — `_initialize_niqe_metric` method (line 440)

## Tasks

<task type="auto">
  <name>Implement NIQE Operator with Lifecycle</name>
  <files>
    - core/operators/niqe.py (NEW)
  </files>
  <action>
    Create `core/operators/niqe.py`:
    
    1. `NiqeOperator` class:
       - Config: name="niqe", category="quality", min=0, max=100
       - `self.model = None` (lazy load)
       - `self.device = None`
    
    2. `initialize(self, config)`:
       - Try import `pyiqa` (handle ImportError gracefully).
       - Detect device: `"cuda" if torch.cuda.is_available() else "cpu"`.
       - Load model: `pyiqa.create_metric("niqe", device=self.device)`.
       - Store in `self.model`.
    
    3. `execute(self, ctx)`:
       - If `self.model is None`: return error "NIQE not initialized".
       - Preprocess: RGB → tensor (CHW, float32, /255).
       - Handle mask if provided (zero out non-subject regions).
       - Run inference.
       - **Normalization**: NIQE returns lower-is-better (typical range 3-50).
         - Invert and scale: `score = max(0, 100 - (raw_niqe * 2))`
         - This maps NIQE 0→100, NIQE 50→0.
       - Return: `{"niqe_score": score}`
    
    4. `cleanup(self)`:
       - Set `self.model = None` to release memory.
  </action>
  <verify>uv run python -c "from core.operators.niqe import NiqeOperator; print(NiqeOperator().config.name)"</verify>
  <done>NIQE operator with full lifecycle implemented</done>
</task>

<task type="auto">
  <name>Add NIQE Tests with Mocking</name>
  <files>
    - tests/unit/test_niqe_operator.py (NEW)
  </files>
  <action>
    Create tests mocking `pyiqa`:
    
    1. Test `initialize`: Verify `pyiqa.create_metric` called with correct args.
    2. Test `execute` with mock model returning fixed value.
    3. Test normalization: mock NIQE=10 → score=80, NIQE=50 → score=0.
    4. Test error handling: uninitialized operator returns error.
    5. Test missing pyiqa: graceful error in initialize.
    
    Use `unittest.mock.patch` for pyiqa dependency.
  </action>
  <verify>uv run pytest tests/unit/test_niqe_operator.py -v</verify>
  <done>Tests pass with mocked model</done>
</task>

## Success Criteria
- [ ] `NiqeOperator` uses `initialize()` for model loading
- [ ] Normalization converts NIQE's lower-is-better to 0-100 scale
- [ ] Graceful fallback if `pyiqa` missing
- [ ] Tests verify lifecycle and execution
