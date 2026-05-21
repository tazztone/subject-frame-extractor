---
name: pr-merger
description: Strategically merge pull requests, handle merge conflicts in test files, and run unit or integration tests. Use when merging pull requests or managing multi-stage branch integrations.
---

# PR Merger

## Quick Start

```bash
git fetch origin && git checkout main && git pull origin main
git merge origin/feature-branch --no-ff
# resolve conflicts if any, then verify tests pass
git push origin main
```

## Workflows

### Single PR Merge

- [ ] `git fetch origin && git checkout main && git pull origin main`
- [ ] `git merge origin/<branch> --no-ff`
- [ ] If conflicts → resolve (see [REFERENCE.md](REFERENCE.md))
- [ ] Run `bash scripts/conflict_scan.sh` to verify no markers remain
- [ ] Run project test suite
- [ ] `git push origin main`
- [ ] Cleanup: `git branch -d <branch> && git push origin --delete <branch>`

### Batch Merge (Multiple PRs)

Order matters. Merge one at a time, verify between each.

1. **Triage** — identify dependency chains and conflict clusters
2. **Order** — foundational/independent PRs first, dependents after
3. **Merge sequentially** — never skip verification between merges

```bash
# Discover overlap
gh pr list --state open
git diff main..origin/<branch> --stat   # per branch
```

- [ ] Sort: fewest conflicts first, dependencies before dependents
- [ ] For each PR in order:
  - [ ] Merge, resolve conflicts, run tests, push
  - [ ] Only proceed to next PR after green tests

### Conflict Resolution (Summary)

| File type | Strategy |
|-----------|----------|
| **Test files** | Union — keep ALL test functions from both sides, deduplicate imports |
| **Production code** | Read both PRs' intent, prefer the more general version, combine if both needed |
| **Config/lock files** | Regenerate from source (`uv lock`, `npm install`, etc.) |

After resolving: `bash scripts/conflict_scan.sh`

See [REFERENCE.md](REFERENCE.md) for detailed patterns.

### Using `gh` CLI

```bash
gh pr diff <number>          # review changes
gh pr checks <number>        # check CI status
gh pr merge <number> --merge --delete-branch  # merge + cleanup
```

### Verification Gate

Discover the project's test command:

1. `ls scripts/*test*` or `grep test Makefile`
2. Check `pyproject.toml` / `package.json` for test scripts
3. Fallback: `pytest` / `npm test` / `make test`

**Never push a merge that hasn't passed tests.**

### Post-Merge Cleanup

- [ ] Delete local branch: `git branch -d <branch>`
- [ ] Delete remote branch: `git push origin --delete <branch>`
- [ ] Verify PR auto-closed, or: `gh pr close <number>`
