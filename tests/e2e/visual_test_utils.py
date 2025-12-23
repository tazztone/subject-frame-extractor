"""
Visual regression testing utilities.

Provides screenshot capture and comparison against baselines using perceptual hashing.
Used to detect unintended UI changes across development cycles.
"""
from pathlib import Path
from typing import Optional
import json
import time

try:
    from PIL import Image
    import imagehash
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False

# Directories for baseline and diff screenshots
BASELINE_DIR = Path(__file__).parent / "baselines"
DIFF_DIR = Path(__file__).parent / "diffs"


def capture_state_screenshot(page, name: str, wait_ms: int = 500) -> Path:
    """
    Capture screenshot of current UI state.
    
    Args:
        page: Playwright page object
        name: Name for the screenshot file
        wait_ms: Time to wait before capture (for animations to settle)
        
    Returns:
        Path to the captured screenshot
    """
    time.sleep(wait_ms / 1000)
    screenshot_path = DIFF_DIR / f"{name}.png"
    screenshot_path.parent.mkdir(exist_ok=True, parents=True)
    page.screenshot(path=str(screenshot_path), full_page=True)
    return screenshot_path


def compare_with_baseline(screenshot_path: Path, threshold: int = 5) -> dict:
    """
    Compare screenshot against baseline using perceptual hash.
    
    Args:
        screenshot_path: Path to the current screenshot
        threshold: Maximum hash distance to consider "same" (0-64 for phash)
        
    Returns:
        Dict with status, diff_score, and action recommendation
    """
    if not HAS_IMAGEHASH:
        return {"status": "skip", "reason": "imagehash not installed"}
    
    baseline_path = BASELINE_DIR / screenshot_path.name
    if not baseline_path.exists():
        return {
            "status": "no_baseline",
            "action": "save_as_baseline",
            "current": str(screenshot_path)
        }
    
    current_hash = imagehash.phash(Image.open(screenshot_path))
    baseline_hash = imagehash.phash(Image.open(baseline_path))
    diff = current_hash - baseline_hash
    
    return {
        "status": "pass" if diff <= threshold else "fail",
        "diff_score": diff,
        "threshold": threshold,
        "baseline": str(baseline_path),
        "current": str(screenshot_path)
    }


def save_as_baseline(screenshot_path: Path) -> Path:
    """
    Promote current screenshot to baseline.
    
    Args:
        screenshot_path: Path to the screenshot to save as baseline
        
    Returns:
        Path to the saved baseline
    """
    BASELINE_DIR.mkdir(exist_ok=True, parents=True)
    baseline_path = BASELINE_DIR / screenshot_path.name
    Image.open(screenshot_path).save(baseline_path)
    return baseline_path


def generate_diff_image(current_path: Path, baseline_path: Path) -> Optional[Path]:
    """
    Generate a visual diff image highlighting differences.
    
    Args:
        current_path: Path to current screenshot
        baseline_path: Path to baseline screenshot
        
    Returns:
        Path to diff image, or None if images are identical
    """
    if not HAS_IMAGEHASH:
        return None
    
    from PIL import ImageChops
    
    current = Image.open(current_path).convert('RGB')
    baseline = Image.open(baseline_path).convert('RGB')
    
    # Resize if dimensions differ
    if current.size != baseline.size:
        baseline = baseline.resize(current.size, Image.Resampling.LANCZOS)
    
    diff = ImageChops.difference(current, baseline)
    
    # Check if there's any difference
    if diff.getbbox() is None:
        return None
    
    # Enhance diff for visibility
    diff_path = DIFF_DIR / f"diff_{current_path.stem}.png"
    diff.save(diff_path)
    return diff_path


def list_baselines() -> list[str]:
    """List all available baseline screenshots."""
    if not BASELINE_DIR.exists():
        return []
    return [p.stem for p in BASELINE_DIR.glob("*.png")]


def cleanup_diffs():
    """Remove all temporary diff screenshots."""
    if DIFF_DIR.exists():
        for f in DIFF_DIR.glob("*.png"):
            f.unlink()
