---
phase: 1
level: 2
researched_at: 2026-02-06
---

# Phase 1 Research: Operator Design

## Questions Investigated

1. How does FiftyOne implement its Operator pattern?
2. What is the minimal protocol needed for analysis operators?
3. How should operators integrate with the existing `AnalysisPipeline`?
4. Which metrics are candidates for migration?

---

## Findings

### FiftyOne Operator Architecture

FiftyOne implements operators using a class-based protocol with these key components:

| Component | Purpose |
|-----------|---------|
| `config` property | Returns `OperatorConfig` with name, label, execution settings |
| `resolve_input(ctx)` | Declares input schema (types, defaults, validation) |
| `execute(ctx)` | Core logic — receives context, returns dict or generator |
| `resolve_output(ctx)` | Declares output schema for result display |

**Key Pattern Insight**: The `ctx` (context) object is the primary interface:
- `ctx.params` — User/caller parameters
- `ctx.view` — Data to process
- `ctx.ops` — Built-in operations (progress, reload)
- `ctx.trigger()` — Call other operators

```python
# FiftyOne operator signature
def execute(self, ctx):
    value = ctx.params["key"]
    view = ctx.view
    return {"result": computed_value}
```

**Sources:**
- [FiftyOne Operator Docs](https://github.com/voxel51/fiftyone/blob/develop/docs/source/plugins/developing_plugins.rst)

---

### Current Metric Implementation Analysis

**Location**: `core/models.py` → `Frame.calculate_quality_metrics()`

**Structure**: 200-line monolithic method with:
- Deeply nested conditionals (`if metrics_to_compute.get(...)`)
- Hardcoded metric logic (Laplacian, Sobel, NIQE, pose estimation)
- Tight coupling to `FrameMetrics`, `QualityConfig`, and `Config`

**Current Metrics**:
| Metric | Type | Dependencies |
|--------|------|--------------|
| `sharpness` | CV | OpenCV Laplacian |
| `edge_strength` | CV | OpenCV Sobel |
| `contrast` | CV | Std deviation |
| `brightness` | CV | Mean pixel |
| `niqe` | ML | PyTorch/pyiqa |
| `eyes_open` | Face | MediaPipe |
| `yaw/pitch/roll` | Face | MediaPipe transform matrix |
| `face_similarity` | Face | InsightFace embedding |

**Problem**: Adding a new metric requires:
1. Editing `FrameMetrics` (add field)
2. Editing `Frame.calculate_quality_metrics()` (add logic)
3. Editing `AnalysisParameters` (add `compute_X` flag)
4. Editing UI (add filter slider)

---

## Decisions Made

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Protocol Style | Python `Protocol` (ABC secondary) | Type hints + duck typing; no forced inheritance |
| Context Pattern | Minimal `OperatorContext` dataclass | Simpler than FiftyOne's full context; we don't need UI resolution |
| Registry Pattern | Dict-based auto-discovery | Scan `core/operators/*.py` for `Operator` implementations |
| Metric Storage | Return dict from `execute()` | Follows FiftyOne pattern; keeps operators stateless |

---

## Proposed Operator Protocol

```python
from typing import Protocol, Any, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class OperatorContext:
    """Minimal context for operator execution."""
    image_rgb: np.ndarray              # Input image (RGB)
    mask: Optional[np.ndarray] = None  # Subject mask (optional)
    config: Optional[Any] = None       # Global Config object
    params: dict = None                # Operator-specific params

@dataclass
class OperatorConfig:
    """Operator metadata and configuration."""
    name: str                          # e.g., "sharpness"
    display_name: str                  # e.g., "Sharpness Score"
    category: str = "quality"          # "quality", "face", "composition"
    default_enabled: bool = True
    requires_mask: bool = False
    requires_face: bool = False

class Operator(Protocol):
    """Protocol for analysis operators."""
    
    @property
    def config(self) -> OperatorConfig:
        """Returns operator metadata."""
        ...
    
    def execute(self, ctx: OperatorContext) -> dict[str, Any]:
        """
        Executes the operator on the given context.
        
        Returns:
            Dict mapping metric names to values (e.g., {"sharpness_score": 85.2})
        """
        ...
```

---

## Patterns to Follow

1. **Stateless Operators**: All state passed via `OperatorContext`; no instance state
2. **Single Responsibility**: One operator = one metric (or tightly related set)
3. **Fail Gracefully**: Return `{"error": str}` on failure, not exceptions
4. **Normalized Scores**: Return 0-100 range for all scores
5. **Lazy Dependencies**: Heavy models (NIQE, MediaPipe) loaded on first use

---

## Anti-Patterns to Avoid

| Anti-Pattern | Why |
|--------------|-----|
| Operator inheritance hierarchies | Protocol-based is simpler |
| Operators modifying input | Stateless = predictable |
| Hard dependencies on `Frame` | Operators work on raw images |
| Global state in operators | Use `ctx.config` for app config |

---

## Dependencies Identified

| Package | Version | Purpose |
|---------|---------|---------|
| (existing) | - | No new external deps for Phase 1 |

---

## Prototype Plan: Sharpness Operator

```python
# core/operators/sharpness.py
import cv2
import numpy as np
from core.operators.base import Operator, OperatorConfig, OperatorContext

class SharpnessOperator:
    @property
    def config(self) -> OperatorConfig:
        return OperatorConfig(
            name="sharpness",
            display_name="Sharpness Score",
            category="quality",
            requires_mask=True,
        )
    
    def execute(self, ctx: OperatorContext) -> dict:
        gray = cv2.cvtColor(ctx.image_rgb, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        if ctx.mask is not None:
            active_mask = ctx.mask > 128
            laplacian = laplacian[active_mask]
        
        variance = float(np.var(laplacian)) if laplacian.size > 0 else 0.0
        
        # Normalize using config scale
        scale = getattr(ctx.config, "sharpness_base_scale", 2500.0)
        score = min(100.0, (variance / scale) * 100.0)
        
        return {"sharpness_score": score}
```

---

## Risks

| Risk | Mitigation |
|------|------------|
| Breaking existing tests | Golden snapshot comparison before/after |
| Performance regression | Operator overhead should be negligible (function call) |
| Migration complexity | Migrate one operator at a time; maintain parallel paths |

---

## Ready for Planning

- [x] Questions answered
- [x] Approach selected  
- [x] Dependencies identified
- [x] Prototype design documented
