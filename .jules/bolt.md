## 2024-05-22 - Vectorizing LPIPS Deduplication
**Learning:** Sequential processing of neural network metrics (like LPIPS) on CPU/GPU is a major bottleneck due to lack of parallelism and overhead.
**Action:** Always batch tensor operations. Refactoring `apply_lpips_dedup` to use batched processing (batch size 32) allows efficient GPU utilization and significantly speeds up the pipeline. Centralizing this logic into `_run_batched_lpips` reduced code duplication.
