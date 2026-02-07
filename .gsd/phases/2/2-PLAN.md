---
phase: 2
plan: 2
wave: 2
---

# Plan 2.2: CLI Resume & Skip Logic

## Objective
Add `--resume` and `--force` flags to the CLI, with logic to skip already-completed runs.

## Context
- .gsd/phases/2/RESEARCH.md
- .gsd/phases/2/1-PLAN.md (fingerprint module)
- cli.py
- core/fingerprint.py

## Tasks

<task type="auto">
  <name>Add Skip Logic to CLI Extract</name>
  <files>cli.py</files>
  <action>
    Modify the `extract` command:
    
    1. Add `--force` flag (default False)
    
    2. At the start, before any expensive work:
       ```python
       from core.fingerprint import load_fingerprint, create_fingerprint, fingerprints_match
       
       existing = load_fingerprint(output_dir)
       if existing and not force:
           new_fp = create_fingerprint(video, extraction_settings)
           if fingerprints_match(new_fp, existing):
               click.secho("✓ Extraction already complete (fingerprint match)", fg="green")
               click.echo(f"   Use --force to re-extract")
               return
       ```
    
    3. Keep the `--clean` flag behavior (delete and re-run).
  </action>
  <verify>python cli.py extract --video "downloads/example clip 720p 2x.mp4" --output ./cli_test_output --nth-frame 5</verify>
  <done>Re-running extraction on existing output shows "already complete" and exits quickly</done>
</task>

<task type="auto">
  <name>Add Resume Flag to CLI Analyze</name>
  <files>cli.py</files>
  <action>
    Modify the `analyze` command:
    
    1. Add `--resume` flag (default False)
    
    2. Pass `resume=resume` to the PreAnalysisEvent:
       ```python
       pre_event = PreAnalysisEvent(
           ...
           resume=resume,
           ...
       )
       ```
    
    3. Also add `--force` flag that ignores existing masks/metadata.
    
    Similarly update the `full` command.
  </action>
  <verify>python cli.py analyze --help | grep -E "(resume|force)"</verify>
  <done>--resume and --force flags appear in analyze help text</done>
</task>

<task type="auto">
  <name>Update Status Command</name>
  <files>cli.py</files>
  <action>
    Enhance the `status` command to show fingerprint info:
    
    1. Load fingerprint if exists
    2. Display fingerprint timestamp and settings hash
    3. Show "Resumable: Yes/No" based on progress.json
    
    Example output:
    ```
    📋 SESSION STATUS: cli_test_output
       ✓ Fingerprint: 2026-02-07T16:25:00 (hash: abc123...)
       ✓ Extraction complete
       ✓ Pre-analysis complete
       ...
       Resumable: Yes (2/2 scenes complete)
    ```
  </action>
  <verify>python cli.py status --session ./cli_test_output</verify>
  <done>Status command shows fingerprint info</done>
</task>

## Success Criteria
- [ ] `python cli.py extract ...` on existing session exits in <2s with skip message
- [ ] `--force` flag forces re-extraction
- [ ] `--resume` flag is available on analyze and full commands
- [ ] Status command shows fingerprint and resume info
