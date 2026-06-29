import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Union

import cv2
import yt_dlp as ytdlp
from yt_dlp.utils import DownloadError

from core.io_utils import validate_video_file
from core.models import Scene

if TYPE_CHECKING:
    from core.config import Config
    from core.events import SessionLoadEvent
    from core.logger import AppLogger

logger = logging.getLogger(__name__)


class MediaSession:
    """Consolidated manager for media lifecycles, YouTube downloads, and session loading."""

    def __init__(
        self,
        config: Optional["Config"] = None,
        source_path: str = "",
        session_path: Optional[str] = None,
        max_resolution: Optional[str] = None,
    ):
        self.config = config
        self.source_path = source_path
        self.session_path = session_path
        self.max_resolution = max_resolution or (self.config.default_max_resolution if self.config else None)
        self.is_youtube = source_path and ("youtube.com/" in source_path or "youtu.be/" in source_path)

    def prepare_video(self, logger: "AppLogger") -> str:
        """Downloads or validates the video source."""
        if not self.source_path:
            raise ValueError("No video source path provided.")

        if self.is_youtube:
            if not self.config:
                raise ValueError("Config required for YouTube downloads.")
            logger.info("Downloading video", component="video", user_context={"source": self.source_path})
            tmpl = self.config.ytdl_output_template
            max_h = None if self.max_resolution == "maximum available" else int(self.max_resolution)
            ydl_opts = {
                "outtmpl": str(Path(self.config.downloads_dir) / tmpl),
                "format": self.config.ytdl_format_string.format(max_res=max_h)
                if max_h
                else "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
                "merge_output_format": "mp4",
                "noprogress": True,
                "quiet": True,
            }
            try:
                from typing import cast

                with ytdlp.YoutubeDL(cast(Any, ydl_opts)) as ydl:
                    info = ydl.extract_info(self.source_path, download=True)
                    return str(Path(ydl.prepare_filename(info)))
            except DownloadError as e:
                raise RuntimeError(f"Download failed: {e}") from e

        local_path = Path(self.source_path)
        validate_video_file(local_path)
        return str(local_path)

    def validate_dir(self, path: Optional[Union[str, Path]] = None) -> tuple[Optional[Path], Optional[str]]:
        p_val = path or self.session_path
        try:
            if p_val is None:
                raise TypeError("path cannot be None")
            p = Path(p_val).expanduser().resolve()
            if p.exists() and p.is_dir():
                self.session_path = str(p)
                return p, None
            return None, f"Session directory does not exist: {p}"
        except Exception as e:
            return None, f"Invalid session path: {e}"

    def get_video_info(self, video_path: Optional[str] = None) -> dict:
        v_path = video_path or self.source_path
        if not v_path:
            return {"width": 0, "height": 0, "fps": 30.0, "frame_count": 0}
        cap = cv2.VideoCapture(str(v_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {v_path}")

        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        cap.release()
        return info

    def load_analysis_scenes(
        self, scenes_data: List[dict], is_folder_mode: bool = False, include_only: bool = True
    ) -> List[Scene]:
        fields = set(Scene.model_fields.keys())
        return [
            Scene(**{k: v for k, v in s.items() if k in fields})
            for s in scenes_data
            if not include_only or is_folder_mode or s.get("status") == "included"
        ]

    def execute_session_load(self, event: "SessionLoadEvent", logger: "AppLogger") -> dict:
        session_path_str = event.session_path or self.session_path
        if not session_path_str or not session_path_str.strip():
            return {"error": "Please enter a path to a session directory."}
        session_path, error = self.validate_dir(session_path_str)
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
                except Exception as e:
                    logger.warning(f"Could not load scene_seeds.json: {e}. Progress for individual scenes may be lost.")
            self.session_path = str(session_path)
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


# Compatibility Shim
VideoManager = MediaSession
