# DECISIONS

> Log of architectural and significant technical decisions.

| Date | Decision | Rationale | Impact |
|------|----------|-----------|--------|
| 2026-02-04 | Pause UI Test Debugging | Sustained timeout in `test_app_flow.py` extraction stage despite selector/mock fixes. Exceeded 3 failures. | Need fresh context to rule out state pollution (GSD Rule 3). |
| 2026-02-06 | Use `importlib` for operator auto-discovery | Avoids manual registry updates when adding operators. Inspired by FiftyOne's plugin pattern. | New operators are discovered automatically from `core/operators/*.py`. |
| 2026-02-06 | Remove `SharpnessOperator` from `__init__.py` exports | Enforce registry-first usage pattern. Discourages direct imports in favor of `OperatorRegistry.get()`. | Breaking change for any code importing `from core.operators import SharpnessOperator`. |
