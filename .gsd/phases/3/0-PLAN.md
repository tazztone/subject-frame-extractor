---
phase: 3
plan: 0
wave: 1
---

# Plan 3.0: Operator Auto-Discovery

## Objective
Implement automatic operator discovery and registration from `core/operators/*.py` files.
This eliminates manual imports in `__init__.py` when adding new operators.

## Context
- `.gsd/SPEC.md` — Project goals
- `core/operators/__init__.py` — Current manual import approach
- `core/operators/registry.py` — Registration mechanisms
- `core/operators/base.py` — Operator Protocol
- `tests/unit/test_operators.py` — Existing test suite

## Tasks

<task type="auto">
  <name>Implement discover_operators() function</name>
  <files>core/operators/registry.py</files>
  <action>
    Add a `discover_operators(package_path: str)` function to `registry.py` that:
    1. Uses `importlib` and `pkgutil` to iterate over modules in a package
    2. Imports each module (triggering `@register_operator` decorators)
    3. Returns a list of discovered operator names
    
    Example pattern:
    ```python
    import importlib
    import pkgutil
    
    def discover_operators(package_path: str = "core.operators") -> list[str]:
        package = importlib.import_module(package_path)
        for importer, modname, ispkg in pkgutil.iter_modules(package.__path__):
            if modname not in ("base", "registry", "__init__"):
                importlib.import_module(f"{package_path}.{modname}")
        return OperatorRegistry.list_names()
    ```
    
    AVOID: Importing base.py or registry.py recursively
  </action>
  <verify>pytest tests/unit/test_operators.py -v -k "discover"</verify>
  <done>New test `test_discover_operators` passes, confirming operators are found.</done>
</task>

<task type="auto">
  <name>Add autodiscovery test</name>
  <files>tests/unit/test_operators.py</files>
  <action>
    Add a new test class `TestAutoDiscovery` with:
    
    1. `test_discover_operators_finds_all`: Calls `discover_operators()` and asserts
       that known operators like "sharpness", "entropy", "niqe" are in the result.
    
    2. `test_discover_operators_skips_base`: Asserts that "base" and "registry" are
       NOT in the discovered operator names (they are infrastructure, not operators).
  </action>
  <verify>pytest tests/unit/test_operators.py::TestAutoDiscovery -v</verify>
  <done>Both tests pass, confirming auto-discovery works correctly.</done>
</task>

## Success Criteria
- [ ] `discover_operators()` function exists in `core/operators/registry.py`
- [ ] 2 new tests pass in `TestAutoDiscovery`
- [ ] No changes needed to individual operator files
