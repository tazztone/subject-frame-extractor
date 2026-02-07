# Phase 7: Hardware-Accelerated Extraction

## Objective
Reduce extraction time for high-resolution and long-duration videos by leveraging GPU hardware acceleration.

## Tasks

### 1. Hardware Detection Utility
- Implement a utility to probe for FFmpeg hardware support:
  - Nvidia (h264_cuvid / hevc_cuvid)
  - Intel/AMD VAAPI
  - macOS VideoToolbox (optional, for cross-platform support)
- Update `Config` to include `ffmpeg_hwaccel` (auto/cuda/vaapi/off).

### 2. Update FFmpeg Pipeline
- Update `run_ffmpeg_extraction` in `core/pipelines.py` to inject `-hwaccel` and relevant codec flags.
- Handle fallback logic: if HW accel fails, automatically retry with CPU.

### 3. Resumable Extraction
- Implement logic to check `frame_map.json` and the `thumbs/` directory before starting.
- Calculate the last successful frame and use `-ss` (seeking) to resume extraction from that point.
- Ensure the progress tracker correctly offsets its starting percentage.

## Success Criteria
- [ ] 4K extraction time reduced by >50% on supported hardware.
- [ ] Pipeline successfully resumes after an intentional interruption.
- [ ] Automatic fallback to CPU if GPU drivers are missing or incompatible.
