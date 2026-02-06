# How to Add a New Operator

This guide explains how to add a new analysis operator (metric) to the Subject Frame Extractor.
The system uses an **auto-discovery** plugin pattern. You just need to create a file in `core/operators/` and register your class.

## Quick Start (Copy-Paste)

Create a new file `core/operators/my_metric.py`:

```python
from core.operators import Operator, OperatorConfig, OperatorContext, OperatorResult
from core.operators import register_operator

@register_operator
class MyMetricOperator:
    @property
    def config(self) -> OperatorConfig:
        return OperatorConfig(
            name="my_metric",
            display_name="My Custom Metric",
            category="quality",
            description="Measures something amazing.",
            min_value=0.0,
            max_value=100.0,
        )

    def execute(self, ctx: OperatorContext) -> OperatorResult:
        # Access image as numpy array (RGB)
        image = ctx.image_rgb
        
        # Compute your score (simulated here)
        score = 85.0 
        
        return OperatorResult(metrics={"my_metric_score": score})
```

That's it! The system will automatically find and load it on startup.

---

## Step-by-Step Guide

### 1. Create the File
Create a new Python file in `core/operators/`. The filename doesn't matter, but should be descriptive (e.g., `exposure.py`, `color_balance.py`).

### 2. Implement the Protocol
Your class must implement the `Operator` protocol:
- **`config` property**: Returns metadata (name, UI label, category).
- **`execute(ctx)` method**: Takes context, returns result.

### 3. Use the Decorator
Add `@register_operator` before your class definition. This registers it with the system.

### 4. Implementation Details

#### The Context Object (`ctx`)
```python
ctx.image_rgb    # np.ndarray: The frame image (RGB)
ctx.mask        # np.ndarray | None: Subject mask (if available)
ctx.config      # Config object: Global app settings
ctx.params      # dict: Any extra params (e.g., face landmarks)
```

#### The Result Object
```python
# Success
return OperatorResult(metrics={"my_score": 95.5})

# Failure (caught safely by pipeline)
return OperatorResult(metrics={}, error="Calculation failed")
```

#### Configuration Options (`OperatorConfig`)
| Field | Description |
|-------|-------------|
| `name` | Unique ID (snake_case). Used in code. |
| `display_name` | Label shown in UI. |
| `default_enabled` | `True` to enable by default. |
| `requires_mask` | `True` if you need `ctx.mask`. |
| `requires_face` | `True` if you need face detection. |

---

## Advanced Topics

### Initialization & Cleanup
If you need to load a heavy model (like a neural net), implement `initialize`.

```python
    def initialize(self, config):
        self.model = load_yolo_model()  # Only called once on startup

    def cleanup(self):
        del self.model  # Called on shutdown
```

### Testing
Add a test file in `tests/unit/test_my_metric.py`.
See `tests/unit/test_operators.py` for examples.

### Example
See `examples/operators/pixel_count.py` for a working example.
