# PR Merger — Reference

## Conflict Resolution

### Test Files (Union Merge)
Test file conflicts are usually additive. Keep **both** sides of the conflict block.
1. Remove conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`).
2. Deduplicate imports.
3. Rename colliding test functions.
4. Run `bash scripts/conflict_scan.sh`.

### Production Code
- **Divergent edits**: Combine intents.
- **Structural moves**: Apply the edit to the new file location.
- **Signature changes**: Keep new signature, reapply body changes.

### Config/Lock Files
Manually merge semantic config (e.g. `pyproject.toml`). **Regenerate** lock files (`uv lock`, `npm install`) instead of manually resolving them.

## Batch Merging
1. **Ordering**: Find overlap (`git diff main..origin/<branch> --name-only`). Merge infrastructure/foundational PRs first.
2. **Cascade Preview**: Preview next merge with `git merge --no-commit --no-ff origin/<branch>`. `git merge --abort` if too messy.
3. **Abort Criteria**: Stop the batch if test failures occur unrelated to conflicts, or if deep architectural conflicts arise.

## Troubleshooting
- **Already merged?**: Run `git fetch origin --prune`.
- **Semantic Conflicts**: If a text-clean merge fails tests, the logic combined incorrectly. Debug the failing test.
