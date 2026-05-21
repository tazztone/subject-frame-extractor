---
name: pr-merger
description: Merge pull requests, resolve conflicts, and run tests. Use when merging PRs or managing multi-branch integrations.
---

# PR Merger

## Quick Start
1. **Prepare**: `git fetch origin && git checkout main && git pull origin main`
2. **Merge**: `git merge origin/<branch> --no-ff`
3. **Resolve**: Handle conflicts if any (see [REFERENCE.md](REFERENCE.md))
4. **Verify**: `bash scripts/conflict_scan.sh` and run tests (e.g. `bash scripts/linux_test_unit.sh`)
5. **Push & Clean**: `git push origin main` and `git push origin --delete <branch>`

## Core Rules
- **Test Files**: Union merge. Keep all tests from both sides.
- **Testing**: NEVER push a merge that fails tests.
- **Batch Merges**: Merge one at a time. Order foundational PRs first. Verify tests pass *between* every merge.

## Helpful Commands
- **List PRs**: `gh pr list --state open`
- **Review PR**: `gh pr diff <number>`
- **Check overlap**: `git diff main..origin/<branch> --stat`
