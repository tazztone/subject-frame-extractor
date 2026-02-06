---
phase: 3
plan: 2
wave: 2
---

# Plan 3.2: "How to Add an Operator" Documentation

## Objective
Write developer documentation explaining how to add new operators.
This completes the Plugin Infrastructure by making it self-documenting.

## Context
- `core/operators/base.py` — Operator Protocol definition
- `core/operators/sharpness.py` — Reference implementation
- `core/operators/__init__.py` — Quick Start docstring

## Tasks

<task type="auto">
  <name>Create HOW_TO_ADD_OPERATOR.md guide</name>
  <files>docs/HOW_TO_ADD_OPERATOR.md</files>
  <action>
    Create a Markdown guide with these sections:
    
    1. **Quick Start**: Minimal example (10 lines) to add a new metric
    2. **Step-by-Step**:
       - Create `core/operators/my_metric.py`
       - Implement `config` property (OperatorConfig)
       - Implement `execute(ctx)` method
       - Use `@register_operator` decorator
    3. **Testing**: How to add tests in `tests/unit/`
    4. **Advanced Topics**:
       - `initialize()`/`cleanup()` for stateful operators
       - `requires_mask` / `requires_face` flags
       - Error handling (return OperatorResult with error)
    5. **Reference**: Link to existing operators as examples
    
    Keep it under 150 lines. Prioritize copy-paste examples.
  </action>
  <verify>test -f docs/HOW_TO_ADD_OPERATOR.md && wc -l docs/HOW_TO_ADD_OPERATOR.md</verify>
  <done>File exists, is under 200 lines, and contains Quick Start section.</done>
</task>

<task type="auto">
  <name>Create dummy operator to validate guide</name>
  <files>core/operators/example.py, tests/unit/test_example_operator.py</files>
  <action>
    Create a simple "PixelCount" operator following the guide exactly:
    
    1. `core/operators/example.py`:
       - Name: "pixel_count"
       - Returns count of non-black pixels as metric
    
    2. `tests/unit/test_example_operator.py`:
       - Test that operator is auto-discovered
       - Test that it returns correct count
    
    This validates the guide is accurate and complete.
  </action>
  <verify>pytest tests/unit/test_example_operator.py -v</verify>
  <done>Example operator discovered and test passes.</done>
</task>

## Success Criteria
- [ ] `docs/HOW_TO_ADD_OPERATOR.md` exists and is comprehensive
- [ ] Example operator works via auto-discovery (no manual import needed)
- [ ] Example operator test passes
