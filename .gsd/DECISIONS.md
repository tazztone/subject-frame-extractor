# DECISIONS

> Log of architectural and significant technical decisions.

| Date | Decision | Rationale | Impact |
|------|----------|-----------|--------|
| 2026-02-04 | Pause UI Test Debugging | Sustained timeout in `test_app_flow.py` extraction stage despite selector/mock fixes. Exceeded 3 failures. | Need fresh context to rule out state pollution (GSD Rule 3). |
