import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

from core.models import Scene

if TYPE_CHECKING:
    from core.events import SessionLoadEvent
    from core.logger import AppLogger


def validate_session_dir(path: Union[str, Path]) -> tuple[Optional[Path], Optional[str]]:
    """Checks if the provided path is a valid session directory."""
    try:
        p = Path(path).expanduser().resolve()
        return (
            p if p.exists() and p.is_dir() else None,
            None if p.exists() and p.is_dir() else f"Session directory does not exist: {p}",
        )
    except Exception as e:
        return None, f"Invalid session path: {e}"


def execute_session_load(event: "SessionLoadEvent", logger: "AppLogger") -> dict:
    """Loads session state from disk."""
    if not event.session_path or not event.session_path.strip():
        return {"error": "Please enter a path to a session directory."}
    session_path, error = validate_session_dir(event.session_path)
    if error or session_path is None:
        return {"error": error or "Invalid session path"}
    config_path, scene_seeds_path, metadata_path = (
        session_path / "run_config.json",
        session_path / "scene_seeds.json",
        session_path / "metadata.db",
    )
    try:
        if not config_path.exists():
            return {"error": f"Could not find 'run_config.json' in {session_path}."}
        try:
            run_config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return {"error": "run_config.json is invalid"}
        scenes_data = []
        scenes_json_path = session_path / "scenes.json"
        if scenes_json_path.exists():
            try:
                scenes_data = [
                    {"shot_id": i, "start_frame": s, "end_frame": e}
                    for i, (s, e) in enumerate(json.loads(scenes_json_path.read_text(encoding="utf-8")))
                ]
            except Exception:
                return {"error": "Failed to read scenes.json"}
        if scene_seeds_path.exists():
            try:
                seeds = {int(k): v for k, v in json.loads(scene_seeds_path.read_text(encoding="utf-8")).items()}
                for s in scenes_data:
                    if s.get("shot_id") in seeds:
                        rec = seeds[s["shot_id"]]
                        rec["best_frame"] = rec.get("best_frame", rec.get("best_seed_frame"))
                        s.update(rec)
                    s.setdefault("status", "included")
            except Exception:
                pass
        return {
            "success": True,
            "session_path": str(session_path),
            "run_config": run_config,
            "scenes": scenes_data,
            "metadata_exists": metadata_path.exists(),
        }
    except Exception as e:
        logger.error(f"Failed to load session: {e}", exc_info=True)
        return {"error": f"Failed to load session: {e}"}


def _load_analysis_scenes(scenes_data: List[dict], is_folder_mode: bool, include_only: bool = True) -> List[Scene]:
    """Converts raw scene data to Scene objects."""
    fields = set(Scene.model_fields.keys())
    return [
        Scene(**{k: v for k, v in s.items() if k in fields})
        for s in scenes_data
        if not include_only or is_folder_mode or s.get("status") == "included"
    ]
