---
phase: 1
plan: 1
wave: 1
---

# Plan 1.1: Operator Protocol Infrastructure

## Objective
Create the foundational `Operator` protocol, `OperatorContext` dataclass, and `OperatorConfig` class that will serve as the contract for all future analysis operators.

## Context
- @.gsd/SPEC.md — Project requirements
- @.gsd/ARCHITECTURE.md — Component structure
- @.gsd/phases/1/RESEARCH.md — Protocol design from research
- @core/models.py — Current `FrameMetrics`, `QualityConfig` patterns

## Tasks

<task type="auto">
  <name>Create operators package structure</name>
  <files>
    - core/operators/__init__.py (NEW)
    - core/operators/base.py (NEW)
  </files>
  <action>
    Create `core/operators/` directory with:
    
    1. `__init__.py`:
       - Export `Operator`, `OperatorContext`, `OperatorConfig`
       - Add package docstring explaining the operator pattern
    
    2. `base.py`:
       - Import typing, dataclasses, numpy, Protocol
       - Define `OperatorConfig` dataclass with fields:
         - `name: str`
         - `display_name: str`
         - `category: str = "quality"` (quality, face, composition)
         - `default_enabled: bool = True`
         - `requires_mask: bool = False`
         - `requires_face: bool = False`
       - Define `OperatorContext` dataclass with fields:
         - `image_rgb: np.ndarray`
         - `mask: Optional[np.ndarray] = None`
         - `config: Optional[Any] = None` (global Config object)
         - `params: dict = field(default_factory=dict)`
       - Define `Operator` Protocol with:
         - `config` property returning `OperatorConfig`
         - `execute(ctx: OperatorContext) -> dict[str, Any]` method
    
    AVOID:
    - DO NOT add any implementation operators yet (that's Plan 1.2)
    - DO NOT import heavy dependencies (torch, cv2) in base.py
    - DO NOT use ABC/abstractmethod (Protocol is sufficient)
  </action>
  <verify>python -c "from core.operators import Operator, OperatorContext, OperatorConfig; print('Import OK')"</verify>
  <done>Imports succeed without errors; Protocol and dataclasses are defined</done>
</task>

<task type="auto">
  <name>Create operator registry skeleton</name>
  <files>
    - core/operators/registry.py (NEW)
  </files>
  <action>
    Create `core/operators/registry.py` with:
    
    1. `OperatorRegistry` class:
       - `_operators: dict[str, Type[Operator]]` class variable
       - `register(cls, operator_class: Type[Operator])` classmethod
       - `get(cls, name: str) -> Optional[Type[Operator]]` classmethod
       - `list_all(cls) -> list[OperatorConfig]` classmethod
       - `discover(cls)` classmethod (placeholder for auto-discovery)
    
    2. `@register_operator` decorator for easy registration
    
    Implementation notes:
    - Keep it simple — dict-based, no metaclass magic
    - `discover()` will scan `core/operators/*.py` but leave as TODO for Phase 3
    - Decorator should call `OperatorRegistry.register(cls)`
    
    AVOID:
    - DO NOT implement auto-discovery yet (deferred to Phase 3)
    - DO NOT add complex validation (keep it lightweight)
  </action>
  <verify>python -c "from core.operators.registry import OperatorRegistry, register_operator; print('Registry OK')"</verify>
  <done>OperatorRegistry class exists with register/get/list_all methods</done>
</task>

<task type="auto">
  <name>Add unit tests for operator infrastructure</name>
  <files>
    - tests/unit/test_operators.py (NEW)
  </files>
  <action>
    Create `tests/unit/test_operators.py` with:
    
    1. `TestOperatorContext`:
       - Test creation with minimal args
       - Test creation with all optional args
       - Test mask field accepts numpy array
    
    2. `TestOperatorConfig`:
       - Test default values are correct
       - Test category options
    
    3. `TestOperatorRegistry`:
       - Test register/get cycle
       - Test list_all returns configs
       - Test get returns None for unknown
    
    4. `TestOperatorProtocol`:
       - Create a mock operator class implementing Protocol
       - Verify it satisfies Protocol check
    
    Use pytest fixtures:
    - `sample_image()` — 100x100 RGB numpy array
    - `sample_mask()` — 100x100 grayscale numpy array
  </action>
  <verify>uv run pytest tests/unit/test_operators.py -v</verify>
  <done>All tests pass; coverage for Protocol, Context, Config, Registry</done>
</task>

## Success Criteria
- [ ] `core/operators/` package exists with `base.py`, `registry.py`, `__init__.py`
- [ ] `Operator` Protocol is importable and documented
- [ ] `OperatorRegistry` supports register/get/list operations
- [ ] Unit tests pass with `uv run pytest tests/unit/test_operators.py`
