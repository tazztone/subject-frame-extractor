---
phase: 1
level: 2
researched_at: 2026-02-04
---

# Phase 1 Research: Testing Taxonomy

## Questions Investigated
1. What is the difference between `tests/integration` and `tests/verification`?
2. How should the test suite be organized for clarity and stability?
3. Where should ad-hoc verification scripts live?

## Findings

### 1. Integration vs Verification
- **Integration (`tests/integration`)**: Contains automated `pytest` suites (`test_real_workflow.py`) that exercise the **real backend pipeline** without mocks (or with minimal mocks). These are formal tests that must pass.
- **Verification (`tests/verification`)**: Contains ad-hoc scripts (e.g., `verify_simple.py`) that connect to a *manually running* server instance (port 7860). These are **manual tools** for debugging, not automated tests.

### 2. E2E (`tests/e2e`)
- Contains automated UI tests using Playwright.
- Crucially, these use a `mock_app` fixture, meaning they test the **UI Logic** but *not* the real backend model execution.

### 3. Root Level Tests
- The root `tests/` folder is cluttered with unit tests (`test_core.py`, `test_utils.py`) mixed with configuration setups.

## Decisions Made
| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Verification Location** | Move to `scripts/verification/` | These are operational tools/scripts, not part of the `pytest` automated suite. |
| **Unit Test Location** | Move to `tests/unit/` | De-clutter the root `tests/` directory. |
| **E2E Naming** | Rename `tests/e2e` → `tests/ui` | Clarifies that these test the UI layer (often mocked), distinct from "Real Integration". |
| **Integration Naming** | Keep `tests/integration` | Standard name for backend pipeline tests. |

## Proposed Structure

```text
subject-frame-extractor/
├── tests/
│   ├── unit/            # Was root test_*.py (Fast, Mocked)
│   ├── integration/     # Real Backend Pipeline (Slow, Real Models)
│   ├── ui/              # Was e2e/ (Playwright + Mock App)
│   ├── conftest.py      # Shared fixtures
│   └── TESTING.md       # The Guide
└── scripts/
    └── verification/    # Was tests/verification/ (Manual checks)
```

## Risks
- **Import Errors**: Moving root tests to `tests/unit/` might break relative imports if they rely on `..`.
- **Mitigation**: Ensure `pytest` is invoked as `python -m pytest` from project root, and check `sys.path` handling in `conftest.py`. Currently `conftest.py` adds project root to sys.path.

## Ready for Planning
- [x] Questions answered
- [x] Approach selected
- [x] Taxonomy defined
