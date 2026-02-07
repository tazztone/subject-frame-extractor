import hashlib
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from datetime import datetime

@dataclass
class RunFingerprint:
    video_path: str
    video_size: int
    video_mtime: float
    extraction_hash: str
    analysis_hash: Optional[str] = None
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

def _hash_dict(data: Dict[str, Any]) -> str:
    """Create a stable hash of a dictionary."""
    # Sort keys to ensure deterministic ordering
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.md5(serialized.encode('utf-8')).hexdigest()

def create_fingerprint(
    video_path: str,
    extraction_settings: Dict[str, Any],
    analysis_settings: Optional[Dict[str, Any]] = None
) -> RunFingerprint:
    """Create a fingerprint for a run."""
    stat = os.stat(video_path)
    
    return RunFingerprint(
        video_path=os.path.abspath(video_path),
        video_size=stat.st_size,
        video_mtime=stat.st_mtime,
        extraction_hash=_hash_dict(extraction_settings),
        analysis_hash=_hash_dict(analysis_settings) if analysis_settings else None
    )

def save_fingerprint(fingerprint: RunFingerprint, output_dir: str) -> None:
    """Save fingerprint to run_fingerprint.json in output directory."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "run_fingerprint.json")
    with open(path, 'w') as f:
        json.dump(asdict(fingerprint), f, indent=2)

def load_fingerprint(output_dir: str) -> Optional[RunFingerprint]:
    """Load fingerprint from run_fingerprint.json if it exists."""
    path = os.path.join(output_dir, "run_fingerprint.json")
    if not os.path.exists(path):
        return None
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return RunFingerprint(**data)
    except (json.JSONDecodeError, TypeError, KeyError):
        return None

def fingerprints_match(new: RunFingerprint, existing: RunFingerprint) -> bool:
    """Check if extraction component of fingerprints match."""
    # Use os.path.abspath to normalize paths
    new_path = os.path.abspath(new.video_path)
    existing_path = os.path.abspath(existing.video_path)
    
    return (
        new_path == existing_path and
        new.video_size == existing.video_size and
        abs(new.video_mtime - existing.video_mtime) < 1.0 and  # Allow 1s slack
        new.extraction_hash == existing.extraction_hash
    )
