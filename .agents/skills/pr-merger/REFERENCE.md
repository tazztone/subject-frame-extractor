# PR Merger — Reference

## Conflict Resolution Strategies

### Test Files (Union Merge)

Test file conflicts are almost always additive — both sides added new tests. The goal is to keep everything.

**Process:**

1. Open the conflicted file and locate all `<<<<<<<` markers
2. For each conflict block, keep **both** sides (remove markers, keep all code)
3. Deduplicate:
   - Imports: keep one copy of each import, union all names
   - Fixtures: if both sides added the same fixture, keep one (prefer the more general version)
   - Test functions: if names collide, one must be renamed
4. Run `bash scripts/conflict_scan.sh` to verify no markers remain
5. Run the test file in isolation to verify: `pytest <file> -v`

**Example — additive test conflict:**

```python
# BEFORE (conflicted)
<<<<<<< HEAD
def test_validates_name():
    assert validate(name="") == Error("name required")
=======
def test_validates_email():
    assert validate(email="bad") == Error("invalid email")
>>>>>>> feature/email-validation

# AFTER (resolved — keep both)
def test_validates_name():
    assert validate(name="") == Error("name required")

def test_validates_email():
    assert validate(email="bad") == Error("invalid email")
```

### Production Code

Production conflicts require understanding *intent*, not just text.

**Process:**

1. Read both PRs' descriptions and the diff context
2. Classify the conflict:
   - **Divergent edits** — both sides changed the same line differently → understand which behavior is correct, or combine
   - **Structural moves** — one side moved/renamed while the other edited → apply the edit to the new location
   - **Signature changes** — one side added a parameter, the other changed the body → merge the signature change, reapply the body change
3. After resolving, verify the module's tests pass in isolation

### Config and Lock Files

Never manually resolve lock files. Regenerate them:

| Tool | Command |
|------|---------|
| uv | `uv lock` |
| pip | `pip freeze > requirements.txt` |
| npm | `rm package-lock.json && npm install` |
| yarn | `rm yarn.lock && yarn install` |
| poetry | `poetry lock` |

For config files (e.g., `pyproject.toml`, `tsconfig.json`): manually merge the semantic content, then regenerate any derived files.

## Batch Merge Strategies

### Dependency-First Ordering

When merging N PRs, build a dependency graph:

1. List files touched by each PR: `git diff main..origin/<branch> --name-only`
2. PRs that touch the same files conflict — merge the *foundational* one first
3. PRs that don't overlap can be merged in any order (prefer smaller first)

**Heuristic ordering:**

1. Infrastructure/config PRs (CI, dependencies, build)
2. Core library/module PRs
3. Feature PRs that build on core changes
4. Test-only PRs (most likely to conflict with everything, easiest to resolve)

### Cascade Conflict Management

After merging PR N, the remaining PRs may have *new* conflicts that didn't exist before. Before merging PR N+1:

```bash
# Preview conflicts without committing
git merge --no-commit --no-ff origin/<next-branch>
git diff --name-only --diff-filter=U   # list conflicted files
git merge --abort                       # abort if too messy
```

This lets you reassess ordering mid-batch.

### Abort Criteria

Stop the batch and reassess if:

- A merge introduces test failures unrelated to conflicts
- More than 3 files have structural (non-additive) conflicts
- Two PRs fundamentally disagree on an API or data model

In these cases, one PR may need to be rebased or reworked before the batch can continue.

## Troubleshooting

### "Already up to date" but PR shows changes

The branch may have been merged already, or the remote ref is stale:

```bash
git fetch origin --prune
git log --oneline main..origin/<branch>   # empty = already merged
```

### Merge succeeds but tests fail

1. Check if the failure exists on either branch independently
2. If both branches pass alone, the conflict is a *semantic* merge conflict — the code merged cleanly but the combined behavior is broken
3. Read the failing test, trace the interaction, fix the integration

### Conflict markers left in binary or generated files

`conflict_scan.sh` catches text files. For generated files, regenerate them from source rather than resolving conflicts.
