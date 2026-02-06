---
phase: 1
plan: 1
wave: 1
---

# Plan 1.1: Operator Protocol Infrastructure

## Objective
Create the foundational `Operator` protocol, `OperatorContext`, `OperatorResult`, and `OperatorConfig` that will serve as the contract for all future analysis operators. Include lifecycle methods for stateful operators and a bridge function skeleton.

## Context
- @.gsd/SPEC.md — Project requirements
- @.gsd/ARCHITECTURE.md — Component structure
- @.gsd/phases/1/RESEARCH.md — Protocol design from research
- @core/models.py — Current `FrameMetrics`, `QualityConfig` patterns

## Tasks

<task type="auto">
  <name>Create operators package with Protocol and types</name>
  <files>
    - core/operators/__init__.py (NEW)
    - core/operators/base.py (NEW)
  </files>
  <action>
    Create `core/operators/` directory with:
    
    1. `base.py` with these components:
    
       a) `OperatorConfig` dataclass:
          - `name: str` — machine name (e.g., "sharpness")
          - `display_name: str` — UI label (e.g., "Sharpness Score")
          - `category: str = "quality"` — (quality, face, composition)
          - `default_enabled: bool = True`
          - `requires_mask: bool = False`
          - `requires_face: bool = False`
          - `min_value: float = 0.0` — for UI sliders
          - `max_value: float = 100.0` — for UI sliders
          - `description: str = ""` — for tooltips
       
       b) `OperatorContext` dataclass:
          - `image_rgb: np.ndarray`
          - `mask: Optional[np.ndarray] = None`
          - `config: Optional[Any] = None` — global Config object
          - `params: dict = field(default_factory=dict)`
       
       c) `OperatorResult` dataclass (formalized return contract):
          - `metrics: dict[str, float]` — computed metric values
          - `error: Optional[str] = None` — error message if failed
          - `warnings: list[str] = field(default_factory=list)`
       
       d) `Operator` Protocol:
          - `config` property returning `OperatorConfig`
          - `execute(ctx: OperatorContext) -> OperatorResult`
          - `initialize(config: Any) -> None` (optional, for model loading)
          - `cleanup() -> None` (optional, for resource cleanup)
    
    2. `__init__.py`:
       - Export all public types
       - Add docstring explaining the operator pattern
    
    AVOID:
    - DO NOT add implementation operators yet
    - DO NOT import cv2/torch in base.py (keep lightweight)
  </action>
  <verify>python -c "from core.operators import Operator, OperatorContext, OperatorConfig, OperatorResult; print('Import OK')"</verify>
  <done>All types importable; Protocol with lifecycle methods defined</done>
</task>

<task type="auto">
  <name>Create operator registry with bridge function</name>
  <files>
    - core/operators/registry.py (NEW)
  </files>
  <action>
    Create `core/operators/registry.py` with:
    
    1. `OperatorRegistry` class:
       - `_operators: dict[str, Operator]` class variable (instances, not types)
       - `_initialized: set[str]` — track which operators have been initialized
       - `register(cls, operator: Operator)` classmethod
       - `get(cls, name: str) -> Optional[Operator]` classmethod
       - `list_all(cls) -> list[OperatorConfig]` classmethod
       - `initialize_all(cls, config: Any)` classmethod — calls initialize() on stateful operators
       - `cleanup_all(cls)` classmethod — calls cleanup() on all operators
    
    2. `@register_operator` decorator:
       - Instantiates the operator class
       - Calls `OperatorRegistry.register(instance)`
    
    3. `run_operators(image_rgb, mask, config, operators=None) -> dict[str, OperatorResult]`:
       - Bridge function skeleton for future pipeline integration
       - If operators is None, run all registered operators
       - Returns dict mapping operator name to OperatorResult
       - Catches exceptions per-operator, doesn't fail entire batch
    
    AVOID:
    - DO NOT implement auto-discovery yet (Phase 3)
  </action>
  <verify>python -c "from core.operators.registry import OperatorRegistry, register_operator, run_operators; print('Registry OK')"</verify>
  <done>Registry with lifecycle management and bridge function exists</done>
</task>

<task type="auto">
  <name>Add comprehensive unit tests</name>
  <files>
    - tests/unit/test_operators.py (NEW)
  </files>
  <action>
    Create `tests/unit/test_operators.py`:
    
    1. Fixtures:
       - `sample_image()` — 100x100 RGB numpy array (random noise)
       - `sample_mask()` — 100x100 grayscale numpy array
       - `mock_config()` — Mock Config with sharpness_base_scale
    
    2. `TestOperatorContext`:
       - Test creation with minimal args
       - Test all optional fields
    
    3. `TestOperatorConfig`:
       - Test default values (min=0, max=100)
       - Test UI metadata fields
    
    4. `TestOperatorResult`:
       - Test success case (metrics only)
       - Test error case (error string set)
       - Test warnings list
    
    5. `TestOperatorRegistry`:
       - Test register/get cycle
       - Test list_all returns configs
       - Test initialize_all/cleanup_all lifecycle
    
    6. `TestRunOperators`:
       - Test bridge function with mock operator
       - Test error isolation (one failure doesn't break others)
    
    7. `TestOperatorProtocol`:
       - Create mock operator implementing Protocol
       - Verify it satisfies typing.runtime_checkable
  </action>
  <verify>uv run pytest tests/unit/test_operators.py -v</verify>
  <done>All tests pass; full coverage for types, registry, bridge</done>
</task>

## Success Criteria
- [ ] `core/operators/` package with `base.py`, `registry.py`, `__init__.py`
- [ ] `OperatorResult` formalizes return contract with error handling
- [ ] `OperatorConfig` includes UI metadata (min/max/description)
- [ ] Lifecycle methods (`initialize`/`cleanup`) in Protocol
- [ ] `run_operators()` bridge function ready for pipeline integration
- [ ] All unit tests pass
