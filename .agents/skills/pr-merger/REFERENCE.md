# PR Merger — Reference

## Review rubric

For each PR diff, check:

- **Correctness** — does the code do what the PR description claims?
- **Tests** — are new behaviours covered? Are existing tests still passing?
- **Breaking changes** — does it change a public API, config format, or DB schema without a migration path?
- **Scope creep** — does the diff contain unrelated changes? Flag them.
- **Security** — any hardcoded secrets, unchecked inputs, or new dependencies with known vulnerabilities?
- **Style** — is it consistent with the surrounding code? (Don't block on nits — comment but approve.)

## Merge method guide

| Situation | Flag | Reason |
|---|---|---|
| Feature branch into main | `--squash` | Clean linear history |
| Release / long-lived branch | `--merge` | Preserves full commit context |
| User requests linear history | `--rebase` | No merge commit |

## Dependency conflicts

If PR A and PR B both touch the same files, merge the smaller/simpler one first, then update the other branch:

```bash
gh pr checkout <larger-pr-number>
git merge main
git push
```
