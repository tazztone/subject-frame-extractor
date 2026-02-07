# DECISIONS

> Log of architectural and significant technical decisions.

| Date | Decision | Rationale | Impact |
|------|----------|-----------|--------|
| 2026-02-04 | Pause UI Test Debugging | Sustained timeout in `test_app_flow.py` extraction stage despite selector/mock fixes. Exceeded 3 failures. | Need fresh context to rule out state pollution (GSD Rule 3). |
| 2026-02-06 | Use `importlib` for operator auto-discovery | Avoids manual registry updates when adding operators. Inspired by FiftyOne's plugin pattern. | New operators are discovered automatically from `core/operators/*.py`. |
| 2026-02-06 | Remove `SharpnessOperator` from `__init__.py` exports | Enforce registry-first usage pattern. Discourages direct imports in favor of `OperatorRegistry.get()`. | Breaking change for any code importing `from core.operators import SharpnessOperator`. |
| 2026-02-07 | Strategic Refinement: CLI-First + Photo Mode MVP | v3.0 goals (extensibility) completed. New focus: testability via CLI, performance via caching, and minimal Photo Mode support. | New milestone with P0 (UI fix, CLI), P1 (caching), P2 (photo MVP). |
| 2026-02-07 | Use fingerprinting for run-skipping | Implemented MD5 hashing of extraction settings + video metadata to detect repeated runs. | Enables "Zero-IO" skip of extraction stage. |
