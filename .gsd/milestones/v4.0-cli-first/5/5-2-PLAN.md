# Plan 5.2: CLI Logging Fix

## Goal
Eliminate duplicate terminal output in CLI mode.

## The Fix
Update `cli.py` line 48:
```python
logger = AppLogger(config, log_dir=output_dir, log_to_file=True, log_to_console=False)
```

## Rationale
- `AppLogger` already supports this parameter.
- CLI uses `click.echo()` for user feedback.
- Logs still go to `run.log` for debugging.

## Verification
```bash
uv run python cli.py extract --video sample.mp4 --output ./test
# Verify: No [INFO]/[DEBUG] lines in terminal
# Verify: ./test/run.log has full logs
```
