# Phase 5: Logging Standardization & UI Decoupling

## Objective
Standardize logging output across all libraries, suppress noise from heavy ML frameworks, and decouple the UI progress queue from the core logging logic.

## Tasks

### 1. Suppress Heavy Framework Noise
- Set environment variables to silence TensorFlow and MediaPipe C++ logs.
- Configure `absl` logging to suppress internal warnings.

### 2. Standardize Output (dictConfig)
- Move to `logging.config.dictConfig` for central management.
- Define a unified format that applies to:
  - `AppLogger` (our code)
  - `sam3`
  - `pyscenedetect`
  - `insightface`
- Ensure color support for CLI is maintained but standardized.

### 3. Decouple UI Progress Queue
- Create a `GradioQueueHandler(logging.Handler)` to handle UI updates.
- Remove `progress_queue` logic from `AppLogger` class.
- The UI will attach this handler at startup; the CLI will not.

## Success Criteria
- [x] No `W0000 00:00...` noise in the terminal.
- [x] `SAM3` and `pyscenedetect` logs match our app's formatting.
- [x] `AppLogger` has zero references to `progress_queue`.
- [x] Both CLI and UI function correctly with the new config.
