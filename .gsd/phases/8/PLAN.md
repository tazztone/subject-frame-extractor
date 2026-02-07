# Phase 8: Proactive Memory & Resiliency

## Objective
Prevent "Out of Memory" (OOM) errors before they occur and ensure the pipeline can recover gracefully from critical failures.

## Tasks

### 1. Dynamic Batch Sizing
- Update `AnalysisPipeline` to monitor memory *before* processing each batch.
- If VRAM usage is > 80%, automatically reduce the batch size by 50%.
- If memory is plentiful, gradually increase batch size back to the user-defined maximum.

### 2. Error Recovery
- Implement a global retry decorator for Operators that catches transient errors (like CUDA timeouts).
- Ensure `metadata.db` is flushed to disk after every batch to prevent data loss on crash.

### 3. Verification & Stress Test
- Run the full pipeline on a 1-hour 4K video.
- Manually trigger OOMs (e.g., by limiting available memory) to verify that dynamic batching kicks in.

## Success Criteria
- [ ] 0 OOM crashes during a stress test on limited-memory hardware (e.g., 8GB GPU).
- [ ] Database integrity maintained after simulated hard crashes.
- [ ] Dynamic batching logic confirmed via logs.
