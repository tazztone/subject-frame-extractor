---
description: Refine the project vision and specification (strategic course correction)
---

# /refine Workflow

<objective>
Refine the core project specification (`SPEC.md`) and reconcile the roadmap based on new learnings or changing requirements. This is a "Reality Check" that confronts Vision vs. Reality.
</objective>

<context>
**Use this workflow when:**
- The project vision has evolved
- "Non-goals" have become goals (or vice versa)
- New constraints or requirements have emerged
- You need to perform a "mid-flight" course correction
- You want to perform a periodic "health check" on project direction

**Do NOT use this workflow for:**
- Minor tweaks to a single phase → Use `/discuss-phase`
- Technical research → Use `/research-phase`
- Fixing bugs or gaps → Use `/debug` or `/plan-milestone-gaps`
</context>

<process>

## 1. Context Sync (Reality Check)

**Mandatory:** Sync understanding with actual codebase state before any strategic discussion.

**Bash:**
```bash
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " GSD ► REFINE: Context Sync"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check Architecture
if [ -f ".gsd/ARCHITECTURE.md" ]; then
    echo ""
    echo "📐 Current Architecture (first 15 lines):"
    head -n 15 ".gsd/ARCHITECTURE.md"
    echo "..."
else
    echo "⚠️  ARCHITECTURE.md not found."
    echo "    Running /map is REQUIRED before strategic refinement."
    echo "    → Run /map now, then return to /refine."
    exit 1
fi

# Check Roadmap Progress
echo ""
echo "🗺️  Current Roadmap Progress:"
grep -E "Status:|Phase [0-9]" ".gsd/ROADMAP.md" 2>/dev/null || echo "No roadmap found."

# Check for blockers/debt
echo ""
echo "📋 Current Blockers/Debt (from TODO.md):"
head -n 10 ".gsd/TODO.md" 2>/dev/null || echo "No TODO.md found."
```

**Decision Point:**
- If architecture is stale → Run `/map` first.
- If roadmap is missing → Run `/new-project` or `/new-milestone`.
- If context looks good → Proceed to Step 2.

---

## 2. Review Current Spec

Display the current finalized specification for review.

**Bash:**
```bash
echo ""
echo "📜 Current SPEC.md:"
echo "───────────────────────────────────────────────────────"
cat ".gsd/SPEC.md"
echo "───────────────────────────────────────────────────────"
```

---

## 3. Strategic Questioning

**Deep Questioning Mode:**

| # | Question | Purpose |
|---|----------|---------|
| 1 | **What has changed?** | Market, Technology, User feedback, Team capacity? |
| 2 | **Verify the Vision** | Is the original vision still accurate and inspiring? |
| 3 | **Challenge the Goals** | Are the current goals still the right ones? Prioritized correctly? |
| 4 | **Review Non-Goals** | Do we need to pull something into scope? Cut something out? |
| 5 | **Reality Check** | Does our architecture actually support this vision? Or is a refactor required? |
| 6 | **Success Criteria** | Are they still measurable? Achievable? Relevant? |

**Action:**
Discuss these points with the user. Do not proceed until a new consensus is reached.

---

## 4. Update SPEC.md

Once the new direction is clear:

1.  **Set Status:** Change `Status: FINALIZED` to `Status: REFINING` temporarily.
2.  **Edit:** Update Vision, Goals, Non-Goals, Constraints, and Success Criteria.
3.  **Re-Finalize:** Change `Status: REFINING` back to `Status: FINALIZED`.

**Bash:**
```bash
# Verify the file is still valid markdown after edits
head -n 10 ".gsd/SPEC.md"
```

---

## 5. Log the Decision (ADR)

**Mandatory:** Major strategic pivots MUST be logged.

Append to `.gsd/DECISIONS.md`:

```markdown
## ADR-{N}: Strategic Refinement - {Date}

**Context:** {Why we are refining}

**Decision:** {What we changed in the Spec}

**Consequences:**
- {Impact on roadmap}
- {Impact on architecture}
- {New technical debt introduced/resolved}
```

---

## 6. Roadmap Reconciliation

A change in Spec usually requires a change in Roadmap.

**Analyze Impact:**

| Spec Change | Roadmap Action |
|-------------|----------------|
| New Goal Added | `/insert-phase` or `/add-phase` |
| Goal Removed | `/remove-phase` |
| Goal Reprioritized | Reorder phases manually |
| Major Scope Change | Consider `/new-milestone` |

**Bash:**
```bash
echo "Current Roadmap:"
cat ".gsd/ROADMAP.md"
```

**Decision:**
- If significant changes needed: Run `/plan` for the affected phases.
- If structural changes needed: Run `/insert-phase` or `/remove-phase`.

---

## 7. Update STATE.md

**Mandatory (Rule 2):** Record the refinement in project memory.

Update `.gsd/STATE.md`:

```markdown
## Current Position
- Phase: {current phase}
- Status: Planning (post-refinement)

## Last Action
Strategic refinement completed. SPEC.md updated with:
- {Change 1}
- {Change 2}

## Next Steps
- {Roadmap reconciliation actions}
- {First phase to re-plan}
```

---

## 8. Commit Changes

Commit the strategic update.

**Bash:**
```bash
git add .gsd/
git commit -m "docs(strategy): refine project specification

- Updated SPEC.md goals/vision
- Logged decision in DECISIONS.md
- Reconciled ROADMAP.md
- Updated STATE.md"
```

---

## 9. Completion

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 GSD ► STRATEGY REFINED ✓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The project specification has been updated.

Files modified:
• .gsd/SPEC.md — Vision/Goals updated
• .gsd/DECISIONS.md — ADR logged
• .gsd/ROADMAP.md — Phases reconciled
• .gsd/STATE.md — Memory updated

───────────────────────────────────────────────────────

▶ NEXT

- /plan {N} — Re-plan affected phases
- /discuss-phase {N} — Clarify new phase scope
- /progress — See the updated path

───────────────────────────────────────────────────────
```

</process>
