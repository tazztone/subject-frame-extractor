# Refactor Plan — Split `app.py` into a Modular `/app/` Package (No Behavior Changes)

> Audience: an autonomous AI refactoring agent.
> Objective: **split a ~2000-line monolith `app.py` into a clean, testable Python package under `/app/` without changing functionality or outputs.**
> Style: deterministic, rule-driven, no creative deviations.

---

## 1) Scope & Non-Goals

**In scope**

* Move existing logic into modules/packages.
* Introduce lightweight interfaces where needed to decouple layers.
* Adjust imports, dependency injection, and wiring so the app runs exactly as before.
* Add minimal scaffolding (package `__init__.py`, `main.py`, and light adapters) to preserve behavior.

**Out of scope**

* No feature additions.
* No algorithmic changes.
* No model/threshold/parameter changes.
* No UI/UX changes.
* No renaming of public-facing behaviors or outputs beyond import paths.

---

## 2) Golden Rules (Do Not Deviate)

1. **Zero functional drift.** Outputs (files, logs, JSONL, thumbnails, masks, metrics, UI behavior) must match pre-refactor results for the same inputs and environment.
2. **Preserve names where feasible.** Function/class names stay the same; only module paths change.
3. **Stable configuration.** Same config values, paths, defaults, and environment checks.
4. **Logging parity.** Keep the same levels, formats, and messages.
5. **Deterministic caching.** Preserve any `lru_cache`/singleton behavior exactly.
6. **No global state expansion.** If globals exist, encapsulate them; do not create new global mutable state.
7. **Atomic steps.** After each phase, the app must run.
8. **Tight imports.** Higher layers may import lower layers; never the reverse (see §5).
9. **Type signatures unchanged.** Public call sites must compile and run with equal semantics.
10. **Idempotent migration.** Running the plan again should not corrupt the repo.

---

## 3) Assumptions (AI May Rely on These)

* The monolith `app.py` currently contains: logging/config setup, dataclasses for domain types, I/O utilities, ML adapters (face/person/grounding/SAM/quality), masking orchestration (seed/propagation), pipelines (extraction/analysis), and a Gradio UI.
* External tool dependencies (FFmpeg, models, CUDA) are already handled in code; keep them unchanged.
* The app is launched via `python app.py` today. After refactor, `python -m app` should be equivalent.

---

## 4) Target Architecture & Directory Layout

```
/app
  /core
    config.py          # Config loading; dirs; derived paths
    logging.py         # Unified logger + SUCCESS level + structured formatter
    types.py           # Shared typed helpers if needed
    utils.py           # safe_execute_with_retry, _to_json_safe, safe_resource_cleanup
    thumb_cache.py     # Thumbnail cache/LRU (if present)
  /domain
    models.py          # Frame, FrameMetrics, Scene, MaskingResult, AnalysisParameters, etc.
  /io
    video.py           # VideoManager, download/ffmpeg helpers
    frames.py          # create_frame_map, rgb_to_pil, render_mask_overlay, etc.
  /ml
    face.py            # get_face_analyzer
    person.py          # PersonDetector, get_person_detector
    grounding.py       # GroundingDINO: load + predict helpers
    sam_tracker.py     # SAM/DAM4SAM tracker init + resolve checkpoints
    quality.py         # NIQE/entropy/other IQA helpers
    downloads.py       # Generic download_model & integrity helpers
  /masking
    seed_selector.py   # SeedSelector (text/face/person/grounding)
    propagate.py       # MaskPropagator
    subject_masker.py  # SubjectMasker (orchestrates seeding + propagation)
  /pipelines
    base.py            # Pipeline base class (params/progress/cancel/logger)
    extract.py         # ExtractionPipeline (scene detection, thumbs)
    analyze.py         # AnalysisPipeline (quality, faces, dedup, JSONL)
  /ui
    app_ui.py          # Gradio UI: tabs, handlers, galleries, histograms
  __init__.py          # Empty (implicit namespace); optional convenience re-exports
  main.py              # Composition root; build UI and launch
```

**Dependency direction (enforced):**

```
ui ──> pipelines ──> masking ──> ml
       │               │
       ├───────────────┘
       └──> io / core / domain
core & domain have no upward imports
```

---

## 5) Module Responsibilities (Concise Map)

* **core.config**: load YAML/config, create dirs, expose derived paths & thresholds.
* **core.logging**: `UnifiedLogger`, custom SUCCESS level, structured formatter, progress queue bridge.
* **core.utils**: generic helpers (`safe_execute_with_retry`, JSON-safe conversions, cleanup).
* **domain.models**: *only* dataclasses/value objects (no heavy logic).
* **io.video**: download, probing, scene detection orchestration, FFmpeg helpers.
* **io.frames**: frame extraction/map creation, color conversions, overlay rendering.
* **ml.face/person/grounding/sam_tracker/quality/downloads**: thin adapters wrapping model loading/prediction and metric computation; keep any `lru_cache` behavior.
* **masking.seed_selector/propagate/subject_masker**: high-level masking; call ML adapters via functions/constructors injected by pipelines.
* **pipelines.base/extract/analyze**: orchestration; receive config/logger/queues; **do not import UI**.
* **ui.app_ui**: Gradio components, events, delegates to pipelines only.
* **main.py**: wire config, logger, queues, UI; provide `build_app()` and `__main__` entry.

---

## 6) Refactor Phases (Deterministic, No Guesswork)

> Perform phases in order. After each phase, run static checks and a smoke run to ensure parity.

### Phase 0 — Prep

* Create `/app/` package with subfolders from §4; add `__init__.py` files.
* Copy `app.py` to a safe backup `app_monolith_snapshot.py` (unchanged).
* Add a temporary `scripts/smoke.py` that imports the current entrypoint and runs a minimal headless path (if possible) to validate equivalence after each phase.

### Phase 1 — Extract Core (Config/Logging/Utils)

1. Move config class & directory constants → `app/core/config.py`.
2. Move logger & formatter & SUCCESS level → `app/core/logging.py`.
3. Move `safe_*` and JSON-safe helpers → `app/core/utils.py`.
4. Update imports within the **monolith** to use `from app.core.config import Config`, etc.
5. Smoke test; expected: identical logs, same config resolution.

### Phase 2 — Extract Domain Models

1. Move dataclasses (`Frame`, `FrameMetrics`, `Scene`, `MaskingResult`, `AnalysisParameters`, etc.) → `app/domain/models.py`.
2. Replace monolith references with `from app.domain.models import Frame, ...`.
3. Smoke test (no behavior change).

### Phase 3 — Extract I/O

1. Move video download/probing/scene/FFmpeg helpers → `app/io/video.py` (`VideoManager`).
2. Move frame utilities (map creation, conversions, overlay) → `app/io/frames.py`.
3. Fix imports; smoke test.

### Phase 4 — Extract ML Adapters

1. Move face/person/grounding/SAM/quality/download logic into `app/ml/*` files one-to-one.
2. Keep function/class names; keep `@lru_cache` wrappers as-is.
3. Ensure any file/model paths still come from `Config` (imported or injected).
4. Fix imports; smoke test (may require the actual environment to load minimally).

### Phase 5 — Extract Masking

1. Move `SeedSelector` → `app/masking/seed_selector.py`.
2. Move `MaskPropagator` → `app/masking/propagate.py`.
3. Move `SubjectMasker` → `app/masking/subject_masker.py`.
4. Adapt their constructors to accept injected adapters if the monolith used direct imports; **default to current implementations** to avoid behavior change.
5. Fix imports; smoke test.

### Phase 6 — Extract Pipelines

1. Create `app/pipelines/base.py` with the base `Pipeline` (params/progress/cancel/logger).
2. Move `ExtractionPipeline` → `app/pipelines/extract.py` (depends on `io` + `core`).
3. Move `AnalysisPipeline` → `app/pipelines/analyze.py` (depends on `masking`, `ml`, `io`, `core`, `domain`).
4. Ensure pipelines receive factories or use module-level functions exactly as before; **no logic edits**.
5. Fix imports; smoke test.

### Phase 7 — Extract UI

1. Move Gradio UI class/functions → `app/ui/app_ui.py`.
2. Ensure UI imports **only** pipelines and `core`/`domain`/`io.frames` for rendering; **never** `ml` directly.
3. Wire events to pipelines as before; keep component IDs/names stable.
4. Smoke test (launch UI) if environment allows.

### Phase 8 — Composition Root

1. Create `app/main.py`:

   * Build `Config` and `UnifiedLogger`.
   * Create `Queue` and `CancelEvent`; inject into `AppUI`.
   * Provide `build_app()` and `if __name__ == "__main__": build_app().launch()`.
2. Replace `app.py` with a thin launcher that calls into `app.main` **or** remove monolith entry and use `python -m app` (pick one; do not keep two public entrypoints).
3. Final smoke test.

---

## 7) Import Rewrite Rules (Mechanical)

Apply these **mechanical** rewrites. If a symbol doesn’t exist in the target module, stop and fail (do not guess).

* `from app import Config` → `from app.core.config import Config`
* `from app import UnifiedLogger` → `from app.core.logging import UnifiedLogger`
* `from app import StructuredFormatter` → `from app.core.logging import StructuredFormatter`
* `from app import (safe_execute_with_retry|_to_json_safe|safe_resource_cleanup)` → `from app.core.utils import ...`
* `Frame|FrameMetrics|Scene|MaskingResult|AnalysisParameters` → `from app.domain.models import ...`
* `VideoManager|scene_*|ffmpeg_*` → `from app.io.video import ...`
* `create_frame_map|rgb_to_pil|render_mask_overlay` → `from app.io.frames import ...`
* `get_face_analyzer` → `from app.ml.face import get_face_analyzer`
* `PersonDetector|get_person_detector` → `from app.ml.person import ...`
* `Grounding*` helpers → `from app.ml.grounding import ...`
* `SAM|DAM4SAM|determine_tracker|tracker init` → `from app.ml.sam_tracker import ...`
* `NIQE|compute_entropy|quality_*` → `from app.ml.quality import ...`
* `download_model` → `from app.ml.downloads import download_model`
* `SeedSelector` → `from app.masking.seed_selector import SeedSelector`
* `MaskPropagator` → `from app.masking.propagate import MaskPropagator`
* `SubjectMasker` → `from app.masking.subject_masker import SubjectMasker`
* `Pipeline` → `from app.pipelines.base import Pipeline`
* `ExtractionPipeline` → `from app.pipelines.extract import ExtractionPipeline`
* `AnalysisPipeline` → `from app.pipelines.analyze import AnalysisPipeline`
* `AppUI` → `from app.ui.app_ui import AppUI`

> If the monolith used relative helpers, search and move the exact symbols; do not create new ones.

---

## 8) Dependency & Injection Rules

* **Pipelines → Masking/ML via factories or module functions.** If today they call `get_face_analyzer()` directly, keep doing so via `app.ml.face`; do not add complexity unless necessary to break cycles.
* **UI → Pipelines only.** UI must not import any `ml.*` modules.
* **Config/Logger** are created once in `main.py` and passed to UI and pipelines. Avoid new global state.
* **Caches** (e.g., `@lru_cache`) remain in adapter modules; do not relocate.

---

## 9) Acceptance Criteria (Parity Tests)

The refactor is **done** only if all pass:

1. **Static parity**

   * `flake8`/`ruff` reports no new errors (ignore pre-existing waivers).
   * `mypy --ignore-missing-imports` unchanged or improved.

2. **Runtime parity**

   * With identical inputs (e.g., a short local video), the following artifacts are byte-for-byte or threshold-equal:

     * Generated thumbnails & masks count and filenames match.
     * JSONL summaries contain the same records/fields/values (ordering can be normalized).
     * Logged lines (levels and text) are equal modulo timestamps.

3. **UI parity**

   * Launches successfully; same tabs and controls; actions trigger the same pipelines and produce the same outputs.

---

## 10) Automation Hints (AST-Safe Moves)

To avoid textual errors, use Python AST for symbol moves:

* Parse `app.py`, build a symbol index (classes, functions, constants).
* For each **target module** in §4–§5, collect exact nodes to extract.
* Write modules with original nodes and original docstrings.
* Rewrite imports using the map in §7.
* Fail if any reference becomes unresolved after the move (run quick import checks).

> Do **not** reformat code beyond import paths (except black/isort if already used in repo).

---

## 11) Minimal New Files (Templates)

**`/app/main.py`**

```python
from queue import Queue
import threading
import shutil
import gradio as gr

from app.core.config import Config
from app.core.logging import UnifiedLogger, StructuredFormatter
from app.ui.app_ui import AppUI

def build_app():
    config = Config()
    logger = UnifiedLogger()
    progress_queue = Queue()
    logger.set_progress_queue(progress_queue)

    if not shutil.which("ffmpeg"):
        with gr.Blocks() as error_app:
            gr.Markdown("# Configuration Error\n\n**FFmpeg not found in PATH.**")
        return error_app

    ui = AppUI(config=config, logger=logger, progress_queue=progress_queue,
               cancel_event=threading.Event())
    return ui.build_ui()

if __name__ == "__main__":
    build_app().launch()
```

**`/app/__init__.py`**

```python
# Intentionally minimal to avoid circular imports.
```

---

## 12) Checklists

**After each phase**

* [ ] All files import without errors: `python -c "import app, importlib; importlib.import_module('app.main')"`
* [ ] Smoke test runs a minimal path without exceptions.
* [ ] Diff of key artifacts (if any) shows parity.

**Before final commit**

* [ ] Single entrypoint (`python -m app`) documented.
* [ ] No residual references to old monolith paths.
* [ ] Caches and model downloads work as before.
* [ ] UI launches and renders expected components.

---

## 13) Deliverables

* The `/app/` package with the structure in §4.
* Updated imports across moved modules.
* `app/main.py` as the composition root.
* A short `REFactor_NOTES.md` summarizing what moved where (one table, no prose).
* (Optional) A `scripts/smoke.py` for local parity checks.

---

## 14) Failure Policy

* If any symbol cannot be confidently placed, **abort and report the exact symbol** and its call sites. Do not invent new modules or APIs.
* If a cyclic import appears, resolve by moving only the **minimum** shared type into `domain.models` or by converting the dependency to a constructor parameter (no logic changes).

---

## 15) Final Instruction to the AI

Perform the phases in §6 exactly, applying the import rewrites in §7 and the dependency rules in §8. Use AST-based extraction, keep names and docstrings, and verify acceptance criteria in §9 after each phase. **Do not modify logic, parameters, thresholds, UI layout, or external behaviors. Do not deviate from this plan.**