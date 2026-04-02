from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING, Any, List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    from insightface.app import FaceAnalysis

    from core.config import Config
    from core.logger import AppLogger
    from core.managers import ModelRegistry, ThumbnailManager

from core.database import Database
from core.enums import SceneStatus
from core.error_handling import ErrorHandler
from core.io_utils import create_frame_map
from core.models import AnalysisParameters, Frame, Scene
from core.operators import OperatorRegistry, run_operators
from core.progress import AdvancedProgressTracker
from core.scene_utils import save_scene_seeds
from core.scene_utils.subject_masker import SubjectMasker
from core.utils import _to_json_safe

from .model_loader import initialize_analysis_models


class Pipeline:
    """Base class for processing pipelines."""

    def __init__(
        self,
        config: "Config",
        logger: "AppLogger",
        params: "AnalysisParameters",
        progress_queue: Queue,
        cancel_event: threading.Event,
    ):
        self.config = config
        self.logger = logger
        self.params = params
        self.progress_queue = progress_queue
        self.cancel_event = cancel_event
        self.error_handler = ErrorHandler(
            self.logger, self.config.retry_max_attempts, self.config.retry_backoff_seconds
        )


def _load_scenes(output_dir: Path) -> List[Scene]:
    """Loads scenes from scenes.json."""
    scenes_path = output_dir / "scenes.json"
    if not scenes_path.exists():
        raise FileNotFoundError("scenes.json not found. Run extraction first.")
    with scenes_path.open("r", encoding="utf-8") as f:
        scenes_data = json.load(f)
    return [Scene(shot_id=i, start_frame=s, end_frame=e) for i, (s, e) in enumerate(scenes_data)]


class PreAnalysisPipeline(Pipeline):
    """Pipeline for pre-analyzing scenes (best frame selection, seeding)."""

    def __init__(
        self,
        config: "Config",
        logger: "AppLogger",
        params: "AnalysisParameters",
        progress_queue: Queue,
        cancel_event: threading.Event,
        thumbnail_manager: "ThumbnailManager",
        model_registry: "ModelRegistry",
        loaded_models: Optional[dict] = None,
    ):
        super().__init__(config, logger, params, progress_queue, cancel_event)
        self.thumbnail_manager = thumbnail_manager
        self.model_registry = model_registry
        self.output_dir = Path(self.params.output_folder)
        self.loaded_models = loaded_models

    def run(self, scenes: List[Scene], tracker: Optional["AdvancedProgressTracker"] = None) -> List[Scene]:
        models = (
            self.loaded_models
            if self.loaded_models
            else initialize_analysis_models(self.params, self.config, self.logger, self.model_registry)
        )
        is_folder_mode = not self.params.video_path
        niqe_metric = self._initialize_niqe_if_needed(models["device"], is_folder_mode)

        masker = SubjectMasker(
            self.params,
            self.progress_queue,
            self.cancel_event,
            self.config,
            face_analyzer=models["face_analyzer"],
            reference_embedding=models["ref_emb"],
            niqe_metric=niqe_metric,
            thumbnail_manager=self.thumbnail_manager,
            logger=self.logger,
            face_landmarker=models["face_landmarker"],
            device=models["device"],
            model_registry=self.model_registry,
            person_detector=models["person_detector"],
        )
        masker.frame_map = masker._create_frame_map(str(self.output_dir))

        previews_dir = self.output_dir / "previews"
        previews_dir.mkdir(exist_ok=True, parents=True)

        for scene in scenes:
            if self.cancel_event.is_set():
                break
            if tracker:
                tracker.step(1, desc=f"Scene {scene.shot_id}")
            self._process_single_scene(scene, masker, previews_dir, is_folder_mode)

        save_scene_seeds(scenes, str(self.output_dir), self.logger)
        return scenes

    def _initialize_niqe_if_needed(self, device: str, is_folder_mode: bool):
        if not is_folder_mode and self.params.pre_analysis_enabled:
            try:
                import pyiqa

                return pyiqa.create_metric("niqe", device=device)
            except ImportError:
                self.logger.debug("pyiqa not installed, NIQE metrics disabled")
            except Exception:
                self.logger.warning("Failed to initialize NIQE metric", exc_info=True)
        return None

    def _process_single_scene(self, scene: Scene, masker: "SubjectMasker", previews_dir: Path, is_folder_mode: bool):
        if is_folder_mode:
            scene.best_frame = scene.start_frame
        elif not scene.best_frame:
            masker._select_best_frame_in_scene(scene, str(self.output_dir))
        if not masker.frame_map:
            return
        fname = masker.frame_map.get(scene.best_frame)
        if not fname:
            return
        thumb_rgb = self.thumbnail_manager.get(self.output_dir / "thumbs" / f"{Path(fname).stem}.webp")
        if thumb_rgb is None:
            return
        bbox, details = masker.get_seed_for_frame(thumb_rgb, seed_config=scene.seed_config or self.params, scene=scene)
        scene.seed_result = {"bbox": bbox, "details": details}
        mask = None
        if bbox and self.params.enable_subject_mask:
            mask = masker.get_mask_for_bbox(thumb_rgb, bbox)
            if mask is not None and mask.size > 0:
                h, w = mask.shape[:2]
                area_pct = (np.sum(mask > 0) / (h * w) * 100) if h * w > 0 else 0.0
                scene.seed_result["details"]["mask_area_pct"] = area_pct
                if not scene.seed_metrics:
                    scene.seed_metrics = {}
                scene.seed_metrics["mask_area_pct"] = area_pct
        from core.image_utils import render_mask_overlay

        overlay_rgb = (
            render_mask_overlay(thumb_rgb, mask, 0.6, logger=self.logger)
            if mask is not None
            else (masker.draw_bbox(thumb_rgb, bbox) if bbox else thumb_rgb)
        )
        import time

        # Add timestamp to bust Gradio's image cache
        preview_path = previews_dir / f"scene_{scene.shot_id:05d}_{int(time.time())}.jpg"
        Image.fromarray(overlay_rgb).save(preview_path)
        scene.preview_path, scene.status = str(preview_path), SceneStatus.INCLUDED


class AnalysisPipeline(Pipeline):
    """Pipeline for analyzing frames (pre-analysis, propagation, full analysis)."""

    def __init__(
        self,
        config: "Config",
        logger: "AppLogger",
        params: "AnalysisParameters",
        progress_queue: Queue,
        cancel_event: threading.Event,
        thumbnail_manager: "ThumbnailManager",
        model_registry: "ModelRegistry",
        loaded_models: Optional[dict] = None,
    ):
        super().__init__(config, logger, params, progress_queue, cancel_event)
        self.output_dir = Path(self.params.output_folder)
        self.db = Database(self.output_dir / "metadata.db", logger=self.logger)
        self.db.error_handler = self.error_handler
        self.thumb_dir = self.output_dir / "thumbs"
        self.masks_dir = self.output_dir / "masks"
        self.processing_lock = threading.Lock()
        self.face_analyzer: Optional["FaceAnalysis"] = None
        self.reference_embedding: Optional[np.ndarray] = None
        self.mask_metadata: dict = {}
        self.face_landmarker: Optional[Any] = None
        self.scene_map: dict = {}
        self.niqe_metric: Optional[Any] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.thumbnail_manager = thumbnail_manager
        self.model_registry = model_registry
        self.loaded_models = loaded_models
        OperatorRegistry.initialize_all(self.config)

    def _initialize_niqe_metric(self):
        if self.niqe_metric is None:
            try:
                import pyiqa

                self.niqe_metric = pyiqa.create_metric("niqe", device=self.device)
            except ImportError:
                self.logger.debug("pyiqa not installed, NIQE metrics disabled")
            except Exception:
                self.logger.warning("Failed to initialize NIQE metric", exc_info=True)

    def run_full_analysis(
        self, scenes_to_process: List[Scene], tracker: Optional["AdvancedProgressTracker"] = None
    ) -> dict:
        if not self.params.video_path:
            return self._run_image_folder_analysis(tracker=tracker)
        progress_file = self.output_dir / "progress.json"
        completed_scene_ids = []
        try:
            if progress_file.exists() and self.params.resume:
                with progress_file.open("r") as f:
                    scenes_to_process = self._filter_completed_scenes(scenes_to_process, json.load(f))
            if not scenes_to_process:
                return {"done": True, "output_dir": str(self.output_dir)}
            self.db.connect()
            self.db.migrate()
            if not self.params.resume:
                self.db.clear_metadata()
            self.scene_map = {s.shot_id: s for s in scenes_to_process}
            models = (
                self.loaded_models
                if self.loaded_models
                else initialize_analysis_models(self.params, self.config, self.logger, self.model_registry)
            )
            self.face_analyzer, self.reference_embedding, self.face_landmarker = (
                models["face_analyzer"],
                models["ref_emb"],
                models["face_landmarker"],
            )
            if self.face_analyzer and self.params.face_ref_img_path:
                self._process_reference_face()
            ext = ".webp" if self.params.thumbnails_only else ".png"
            masker = SubjectMasker(
                self.params,
                self.progress_queue,
                self.cancel_event,
                self.config,
                create_frame_map(self.output_dir, self.logger, ext=ext),
                self.face_analyzer,
                self.reference_embedding,
                thumbnail_manager=self.thumbnail_manager,
                niqe_metric=self.niqe_metric,
                logger=self.logger,
                face_landmarker=self.face_landmarker,
                device=models["device"],
                model_registry=self.model_registry,
            )
            self.mask_metadata = masker.run_propagation(str(self.output_dir), scenes_to_process, tracker=tracker)
            completed_scene_ids = [s.shot_id for s in scenes_to_process]
            if self.cancel_event.is_set():
                return {"log": "Propagation cancelled.", "done": False}
            return {"done": True, "output_dir": str(self.output_dir)}
        except Exception as e:
            self.logger.error("Propagation failed", exc_info=True)
            return {"error": str(e), "done": False}
        finally:
            self._save_progress_bulk(completed_scene_ids, progress_file)

    def run_analysis_only(
        self, scenes_to_process: List[Scene], tracker: Optional["AdvancedProgressTracker"] = None
    ) -> dict:
        try:
            self.db.connect()
            self.db.migrate()
            if not self.params.resume:
                self.db.clear_metadata()
            self.scene_map = {s.shot_id: s for s in scenes_to_process}
            models = (
                self.loaded_models
                if self.loaded_models
                else initialize_analysis_models(self.params, self.config, self.logger, self.model_registry)
            )
            self.face_analyzer, self.reference_embedding, self.face_landmarker = (
                models["face_analyzer"],
                models["ref_emb"],
                models["face_landmarker"],
            )
            if self.face_analyzer and self.params.face_ref_img_path:
                self._process_reference_face()
            mask_metadata_path = self.output_dir / "mask_metadata.json"
            if mask_metadata_path.exists():
                with open(mask_metadata_path, "r") as f:
                    self.mask_metadata = {Path(k).stem: v for k, v in json.load(f).items()}
            if tracker:
                tracker.set_stage("Analyzing frames")
            self._initialize_niqe_metric()
            metrics_to_compute = {
                k: getattr(self.params, f"compute_{k}", False)
                for k in [
                    "quality",
                    "sharpness",
                    "edge_strength",
                    "contrast",
                    "brightness",
                    "entropy",
                    "eyes_open",
                    "yaw",
                    "pitch",
                ]
            }
            self._run_analysis_loop(scenes_to_process, metrics_to_compute, tracker=tracker)
            self.db.flush()
            if self.cancel_event.is_set():
                return {"log": "Analysis cancelled.", "done": False}
            return {"done": True, "output_dir": str(self.output_dir)}
        except Exception as e:
            self.logger.error("Analysis failed", exc_info=True)
            return {"error": str(e), "done": False}

    def _filter_completed_scenes(self, scenes: List[Scene], progress_data: Optional[dict]) -> List[Scene]:
        if not progress_data:
            return scenes
        completed = set(progress_data.get("completed_scenes", []))
        return [s for s in scenes if s.shot_id not in completed]

    def _save_progress_bulk(self, completed_scene_ids: List[int], progress_file: Path):
        if not completed_scene_ids:
            return
        progress_data = {"completed_scenes": []}
        if progress_file.exists():
            try:
                with progress_file.open("r") as f:
                    progress_data = json.load(f) or {}
            except Exception:
                self.logger.warning(
                    f"Failed to read progress file {progress_file}, it may be corrupted.", exc_info=True
                )
        if not isinstance(progress_data, dict):
            progress_data = {}
        progress_data["completed_scenes"] = sorted(
            list(set(progress_data.get("completed_scenes", [])) | set(completed_scene_ids))
        )
        try:
            with progress_file.open("w") as f:
                json.dump(progress_data, f)
        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")

    def _process_reference_face(self):
        if not self.face_analyzer:
            self.logger.error("Face analyzer not initialized before processing reference face")
            return

        ref_path = Path(self.params.face_ref_img_path)
        if not ref_path.is_file():
            raise FileNotFoundError(f"Ref face not found: {ref_path}")
        ref_img = cv2.imread(str(ref_path))
        if ref_img is None:
            raise ValueError("Could not read ref image")
        ref_faces = self.face_analyzer.get(ref_img)
        if not ref_faces:
            raise ValueError("No face in ref image")
        self.reference_embedding = max(ref_faces, key=lambda x: x.det_score).normed_embedding

    def _run_image_folder_analysis(self, tracker: Optional["AdvancedProgressTracker"] = None) -> dict:
        scenes = _load_scenes(self.output_dir)
        scenes_to_process = [s for s in scenes if s.status == SceneStatus.INCLUDED or not self.params.resume]
        if tracker:
            tracker.start(len(scenes_to_process), desc="Analyzing Images")
        return self.run_analysis_only(scenes_to_process, tracker=tracker)

    def _run_analysis_loop(
        self,
        scenes_to_process: List[Scene],
        metrics_to_compute: dict,
        tracker: Optional["AdvancedProgressTracker"] = None,
    ):
        frame_map = create_frame_map(self.thumb_dir.parent, self.logger)
        all_nums = sorted(
            list(
                {
                    fn
                    for s in scenes_to_process
                    for fn in range(s.start_frame, s.end_frame if s.end_frame > s.start_frame else s.start_frame + 1)
                    if fn in frame_map
                }
            )
        )
        image_files = [self.thumb_dir / frame_map[fn] for fn in all_nums]
        num_workers = (
            1 if self.params.disable_parallel else min(os.cpu_count() or 4, self.config.analysis_default_workers)
        )
        current_bs = self.config.analysis_default_batch_size
        processed = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            while processed < len(image_files) or futures:
                if (
                    torch.cuda.is_available()
                    and torch.cuda.memory_reserved(0) / torch.cuda.get_device_properties(0).total_memory > 0.85
                ):
                    current_bs = max(1, current_bs // 2)
                elif current_bs < self.config.analysis_default_batch_size:
                    current_bs = min(self.config.analysis_default_batch_size, current_bs + 2)
                while len(futures) < num_workers and processed < len(image_files):
                    end = min(processed + current_bs, len(image_files))
                    batch_paths = image_files[processed:end]
                    fut = executor.submit(self._process_batch, batch_paths, metrics_to_compute)
                    setattr(fut, "_batch_len", len(batch_paths))
                    futures.append(fut)
                    processed = end
                done = [f for f in futures if f.done()]
                futures = [f for f in futures if not f.done()]
                for f in done:
                    # Determine batch size for this future to update tracker correctly even on failure
                    batch_len = getattr(f, "_batch_len", current_bs)
                    try:
                        f.result()
                    except Exception as e:
                        self.logger.error(f"Batch error: {e}")
                    finally:
                        if tracker:
                            tracker.step(batch_len)
                if self.cancel_event.is_set():
                    break
                if futures:
                    from concurrent.futures import FIRST_COMPLETED, wait

                    wait(futures, timeout=0.1, return_when=FIRST_COMPLETED)
                else:
                    time.sleep(0.01)

    def _process_batch(self, paths: List[Path], metrics: dict) -> int:
        # Pre-load all images in the batch to avoid repeated disk I/O in the loop
        preloaded_data = []
        for p in paths:
            if self.cancel_event.is_set():
                break
            img = self.thumbnail_manager.get(p)
            if img is not None:
                mask_meta = self.mask_metadata.get(p.stem, {})
                mask_thumb = None
                if mask_meta.get("mask_path"):
                    mp = Path(mask_meta["mask_path"])
                    if not mp.is_absolute():
                        mp = self.masks_dir / mp.name
                    if mp.exists():
                        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                        if m is not None:
                            mask_thumb = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

                preloaded_data.append({"path": p, "img": img, "mask_thumb": mask_thumb, "mask_meta": mask_meta})

        for data in preloaded_data:
            self._process_single_frame(data["path"], metrics, preloaded=data)
        return len(paths)

    def _process_single_frame(self, path: Path, metrics: dict, preloaded: Optional[dict] = None):
        if self.cancel_event.is_set():
            return
        match = re.search(r"frame_(\d+)", path.name)
        if not match:
            return
        try:
            if preloaded:
                img = preloaded["img"]
                mask_thumb = preloaded["mask_thumb"]
                mask_meta = preloaded["mask_meta"]
            else:
                img = self.thumbnail_manager.get(path)
                if img is None:
                    return
                mask_meta = self.mask_metadata.get(path.stem, {})
                mask_thumb = None
                if mask_meta.get("mask_path"):
                    mp = Path(mask_meta["mask_path"])
                    if not mp.is_absolute():
                        mp = self.masks_dir / mp.name
                    if mp.exists():
                        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                        if m is not None:
                            mask_thumb = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            frame = Frame(image_data=img, frame_number=-1)
            meta = {"filename": path.name, "metrics": {}}
            faces, bbox = None, None
            if self.params.compute_face_sim and self.face_analyzer:
                with self.processing_lock:
                    try:
                        faces = self.face_analyzer.get(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        if faces:
                            bbox = max(faces, key=lambda x: x.det_score).bbox.astype(int)
                    except Exception:
                        self.logger.warning("Face analysis failed for frame", exc_info=True)

            if any(metrics.values()) or self.params.compute_niqe:
                res = run_operators(
                    image_rgb=img,
                    mask=mask_thumb,
                    config=self.config,
                    model_registry=self.model_registry,
                    logger=self.logger,
                    params={
                        "face_bbox": bbox,
                        "faces": faces,
                        "reference_embedding": self.reference_embedding,
                        "face_landmarker": self.face_landmarker,
                        "mask_meta": mask_meta,
                    },
                )
                for name, r in res.items():
                    if r.success:
                        for k, v in r.metrics.items():
                            if k == "face_sim":
                                frame.face_similarity_score = v
                            elif k == "face_conf":
                                frame.max_face_confidence = v
                            elif hasattr(frame.metrics, k):
                                setattr(frame.metrics, k, v)
                        if r.data and "phash" in r.data:
                            meta["phash"] = r.data["phash"]
            meta["metrics"] = frame.metrics.model_dump()
            if self.params.compute_face_sim:
                if frame.face_similarity_score is not None:
                    meta["face_sim"] = frame.face_similarity_score
                if frame.max_face_confidence is not None:
                    meta["face_conf"] = frame.max_face_confidence
            if self.params.compute_subject_mask_area:
                meta.update(mask_meta)
            if meta.get("shot_id") is not None and (s := self.scene_map.get(meta["shot_id"])) and s.seed_metrics:
                meta["seed_face_sim"] = s.seed_metrics.get("best_face_sim")
            if hasattr(self.params, "dedup_thresh"):
                meta["dedup_thresh"] = self.params.dedup_thresh  # type: ignore
            if frame.error:
                meta["error"] = frame.error
            if meta.get("mask_path"):
                meta["mask_path"] = Path(meta["mask_path"]).name
            self.db.insert_metadata(_to_json_safe(meta))
        except Exception as e:
            self.db.insert_metadata({"filename": path.name, "error": f"failed: {e}", "error_severity": "ERROR"})
