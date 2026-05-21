---
name: pr-merger
description: Strategically merge pull requests, handle merge conflicts in test files, and run unit or integration tests. Use when merging pull requests or managing multi-stage branch integrations.
---

# PR Merger

## Workflows

### 1. Single PR Merge (Git or CLI)

Choose a method to merge into `main`:

**Method A: GitHub CLI (Cleanest)**
```bash
gh pr diff <num>                             # Review changes
gh pr checks <num>                           # Confirm CI passes
gh pr merge <num> --merge --delete-branch    # Merge & cleanup remote
```

**Method B: Git CLI (For Conflict Handling)**
```bash
git fetch origin && git checkout main && git pull origin main
git merge origin/<branch> --no-ff
# If conflicts -> resolve (see REFERENCE.md)
bash scripts/conflict_scan.sh                # Ensure no conflict markers remain
```

- [ ] Run project test suite (e.g., `pytest`, `npm test`)
- [ ] `git push origin main`
- [ ] Cleanup branch: `git branch -d <branch> && git push origin --delete <branch> 2>/dev/null || true`

---

### 2. Sequential Batch Merges

To merge multiple PRs, order them to minimize cascade conflicts:

1. **Sort**: Touch foundational/infrastructure PRs first, feature PRs second, test-only PRs last.
2. **Sequential Merge**: Merge, resolve conflicts, test, and push **one PR at a time**. Never stack unresolved merges.
3. **Cascade Check**: Before merging next, run `git merge --no-commit --no-ff origin/<next-branch>` to preview conflicts.

---

### 3. Conflict Resolution Summary

| File Type | Resolution Strategy |
| :--- | :--- |
| **Test Files** | **Union Merge**: Keep all test cases/functions from both sides; deduplicate imports. |
| **Source Code** | **Semantic Merge**: Combine logic carefully; prefer more general/robust APIs. |
| **Lock Files** | **Regenerate**: Discard conflicts and regenerate (`uv lock`, `npm install`, etc.). |

*For detailed patterns and abort criteria, see [REFERENCE.md](REFERENCE.md).*

---

### 4. Verification Gate

Before pushing any merge, locate and run the test suite:
1. Scan for test scripts: `ls scripts/*test*` or check `pyproject.toml` / `package.json`.
2. Run tests to verify the integration. **Never push untested merges.**
