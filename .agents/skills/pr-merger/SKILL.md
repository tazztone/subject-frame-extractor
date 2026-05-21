---
name: pr-merger
description: Strategically merge pull requests, handle merge conflicts in test files, and run unit or integration tests. Use when merging pull requests or managing multi-stage branch integrations.
---

# Strategic PR Merging & Integration

## Quick start

Check out a PR branch, merge `main`, verify tests, and merge back to `main`:
```bash
# Checkout and merge main
gh pr checkout <PR_NUMBER>
git merge main

# Verify locally
uv run pytest tests/unit/test_events.py

# Push and merge
git push origin <BRANCH_NAME>
gh pr merge <PR_NUMBER> --merge --delete-branch
```

---

## Strategic Integration Workflows

When handling multiple concurrent PRs, group them in dependency order:
1. **Phase 1: Security & Independent PRs** (e.g. database security patches, code-health imports).
2. **Phase 2: Functional Core Updates** (functional pipeline or utility updates).
3. **Phase 3: Shared Test File Integrations** (unit test suites with high merge conflict risks).

---

## Conflict Resolution Protocols

When merging `main` into a PR branch triggers a conflict in shared test files (e.g., `tests/unit/test_events.py` or `tests/unit/test_progress.py`):
1. **Preserve Both Test Cases**: Do not select one over the other. Extract and merge unique test cases (e.g., both `test_validate_output_folder_direct` and `test_strip_emoji_from_strategy`).
2. **Handle Mocking vs Real Files**: Favor test methods using `tmp_path` or real fixtures over older generic mocks.
3. **Commit Stage**: Stage files with `git add`, run `git commit`, and handle formatting checks.

---

## Verification & Execution Protocols

Always run the correct commands depending on the tests to verify the integrations:

### 1. Unit Tests (Fast, No GPU)
```bash
bash scripts/linux_test_unit.sh
```
*   **Coverage Target**: Must maintain or exceed the **80%** repository coverage threshold.

### 2. GPU / Integration / End-to-End Tests
```bash
export PYTEST_INTEGRATION_MODE=true
uv run --no-sync pytest tests/integration/ -m "integration or gpu_e2e" -n 1 -v --no-cov
```
> [!IMPORTANT]
> - Always use `export PYTEST_INTEGRATION_MODE=true` to properly propagate global mock disabling to sub-workers.
> - Always use `-n 1` for GPU tests to prevent VRAM exhaustion and out-of-memory crashes.

### 3. Handling Transient Flakes
If a commit fails due to known parallel/xdist flakes in pre-commit hooks (e.g., `test_create_scene_thumbnail_svg`), and individual targeted tests pass successfully, bypass the pre-commit pytest check using the `--no-verify` flag during commit:
```bash
git commit --no-verify -m "Merge branch 'main' into PR <PR_NUMBER>"
```
