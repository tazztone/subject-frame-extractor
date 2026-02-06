---
phase: 3
plan: 0
wave: 1
---

# Plan 3.0: Operator Auto-Discovery & Refactor

## Objective
Implement automatic operator discovery and replace manual imports in `__init__.py`.
After this, adding a new operator requires only creating the file — no imports needed.

## Context
- `core/operators/__init__.py` — Current manual import approach
- `core/operators/registry.py` — Registration mechanisms
- `tests/unit/test_operators.py` — Existing test suite

## Breaking Change Notice

> [!WARNING]
> This plan removes individual operator class exports from `core.operators`.
> Code like `from core.operators import SharpnessOperator` will break.
> Users should use `OperatorRegistry.get("sharpness")` instead.

## Tasks

<task type="auto">
  <name>Implement discover_operators() function</name>
  <files>core/operators/registry.py</files>
  <action>
    Add a `discover_operators(package_path: str)` function to `registry.py` that:
    1. Uses `importlib` and `pkgutil` to iterate over modules in a package
    2. Imports each module (triggering `@register_operator` decorators)
    3. Returns a list of discovered operator names
    
    Implementation:
    ```python
    import importlib
    import pkgutil
    
    def discover_operators(package_path: str = "core.operators") -> list[str]:
        """Auto-discover and register operators from a package."""
        package = importlib.import_module(package_path)
        for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
            if modname not in ("base", "registry"):
                importlib.import_module(f"{package_path}.{modname}")
        return OperatorRegistry.list_names()
    ```
    
    Also add to module exports.
  </action>
  <verify>python -c "from core.operators.registry import discover_operators; print(discover_operators())"</verify>
  <done>Function returns list containing at least ["sharpness", "entropy", "niqe"].</done>
</task>

<task type="auto">
  <name>Refactor __init__.py to use auto-discovery</name>
  <files>core/operators/__init__.py</files>
  <action>
    1. Remove ALL individual operator imports (SharpnessOperator, etc.)
    2. Remove individual operators from `__all__`
    3. Add call to `discover_operators()` at module load
    4. Keep only framework type exports
    
    New `__init__.py` should look like:
    ```python
    from core.operators.base import (
        Operator, OperatorConfig, OperatorContext, OperatorResult,
    )
    from core.operators.registry import (
        OperatorRegistry, register_operator, run_operators, discover_operators,
    )
    
    # Auto-discover all operators in this package
    discover_operators()
    
    __all__ = [
        "Operator", "OperatorConfig", "OperatorContext", "OperatorResult",
        "OperatorRegistry", "register_operator", "run_operators", "discover_operators",
    ]
    ```
  </action>
  <verify>pytest tests/unit/test_operators.py -v --tb=short</verify>
  <done>All existing tests pass (operators discovered automatically).</done>
</task>

<task type="auto">
  <name>Add auto-discovery tests</name>
  <files>tests/unit/test_operators.py</files>
  <action>
    Add `TestAutoDiscovery` class with:
    
    1. `test_discover_finds_known_operators`: Assert "sharpness", "entropy", "niqe" discovered
    2. `test_discover_skips_infrastructure`: Assert "base" and "registry" NOT in list
    
    Note: Must use a fresh registry for these tests (fixture clears it).
  </action>
  <verify>pytest tests/unit/test_operators.py::TestAutoDiscovery -v</verify>
  <done>Both new tests pass.</done>
</task>

## Success Criteria
- [ ] `discover_operators()` function works and returns known operators
- [ ] `__init__.py` has no manual operator imports
- [ ] All 28+ existing tests pass
- [ ] 2 new auto-discovery tests pass
