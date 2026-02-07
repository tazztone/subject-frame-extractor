#!/usr/bin/env python3
"""
CLI for Subject Frame Extractor

A command-line interface for headless operation, automated testing, and scripting.

Usage:
    python cli.py extract --video path/to/video.mp4 --output ./results
    python cli.py analyze --session ./results --face-ref path/to/face.png
    python cli.py full --video path/to/video.mp4 --output ./results

For help on any command:
    python cli.py <command> --help
"""

import json
import shutil
import sys
import threading
from collections import deque
from pathlib import Path
from queue import Queue

import click
import torch

from core.config import Config
from core.events import ExtractionEvent, PreAnalysisEvent, PropagationEvent
from core.logger import AppLogger, setup_logging
from core.managers import ModelRegistry, ThumbnailManager
from core.pipelines import (
    execute_analysis,
    execute_extraction,
    execute_pre_analysis,
    execute_propagation,
)
from core.filtering import apply_all_filters_vectorized
from core.database import Database


def _setup_runtime(output_dir: Path, verbose: bool = False):
    """Initialize shared runtime components."""
    config = Config()
    config.monitoring_memory_critical_threshold_mb = 64000  # Disable watchdog for CLI
    config.log_level = "DEBUG" if verbose else "INFO"

    progress_queue = Queue()
    setup_logging(config, log_dir=output_dir, log_to_console=True, progress_queue=None)
    logger = AppLogger(config)
    
    cancel_event = threading.Event()
    model_registry = ModelRegistry(logger)
    thumbnail_manager = ThumbnailManager(logger, config)

    return config, logger, progress_queue, cancel_event, model_registry, thumbnail_manager


def _run_pipeline(generator, stage_name: str) -> dict:
    """Consume a pipeline generator and return the final result."""
    result = None
    for update in generator:
        if isinstance(update, dict):
            result = update
            # Print progress if available
            if "unified_log" in update:
                click.echo(f"  {update['unified_log']}")
    
    if not result or not result.get("done"):
        click.secho(f"❌ {stage_name} failed.", fg="red")
        if result:
            click.echo(f"   Error: {result.get('unified_log', 'Unknown error')}")
        sys.exit(1)
    
    click.secho(f"✓ {stage_name} complete.", fg="green")
    return result


@click.group()
@click.version_option(version="4.0.0", prog_name="Subject Frame Extractor CLI")
def cli():
    """Subject Frame Extractor - CLI for headless operation."""
    pass


@cli.command()
@click.option("--source", "-s", required=True, type=click.Path(exists=True), help="Path to input video file or image folder.")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output directory for extracted frames.")
@click.option("--method", "-m", default="every_nth_frame", type=click.Choice(["every_nth_frame", "all", "keyframes"]), help="Extraction method (ignored for folders).")
@click.option("--nth-frame", "-n", default=3, type=int, help="Extract every Nth frame (for every_nth_frame method).")
@click.option("--max-resolution", "-r", default="1080", type=click.Choice(["480", "720", "1080", "1440", "2160"]), help="Max video resolution.")
@click.option("--thumb-mp", default=0.5, type=float, help="Thumbnail megapixels.")
@click.option("--scene-detect/--no-scene-detect", default=True, help="Enable scene detection (ignored for folders).")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
@click.option("--clean", is_flag=True, help="Remove existing output directory before extraction.")
@click.option("--force", is_flag=True, help="Force extraction even if fingerprint matches.")
def extract(source, output, method, nth_frame, max_resolution, thumb_mp, scene_detect, verbose, clean, force):
    """
    Extract frames from a video file or ingest an image folder.
    
    This is the first step in the pipeline. It extracts thumbnails from the source
    and prepares it for analysis.
    """
    output_dir = Path(output)
    source_path = Path(source)
    is_video = not source_path.is_dir()
    
    if clean and output_dir.exists():
        click.echo(f"🧹 Cleaning existing output: {output_dir}")
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_video:
        # Check for existing fingerprint
        from core.fingerprint import load_fingerprint, create_fingerprint, fingerprints_match
        
        ext_settings = {
            "method": method,
            "nth_frame": nth_frame,
            "max_resolution": max_resolution,
            "scene_detect": scene_detect,
            "thumb_megapixels": thumb_mp,
        }
        
        existing_fp = load_fingerprint(str(output_dir))
        if existing_fp and not force:
            try:
                new_fp = create_fingerprint(str(source), ext_settings)
                if fingerprints_match(new_fp, existing_fp):
                    click.secho("\n🔎 Fingerprint match detected!", fg="yellow")
                    click.secho("✓ Extraction already complete (use --force to re-run)", fg="green")
                    return
            except Exception as e:
                click.secho(f"⚠️ Could not verify fingerprint: {e}", fg="yellow")

    click.secho(f"\n🎬 {'EXTRACTION' if is_video else 'INGESTION'}", fg="cyan", bold=True)
    click.echo(f"   Source: {source}")
    click.echo(f"   Output: {output_dir}")
    if is_video:
        click.echo(f"   Method: {method} (every {nth_frame} frames)")
    
    config, logger, progress_queue, cancel_event, model_registry, thumbnail_manager = _setup_runtime(output_dir, verbose)
    
    event = ExtractionEvent(
        source_path=str(source),
        method=method,
        interval="1.0",
        nth_frame=nth_frame,
        max_resolution=max_resolution,
        thumbnails_only=True,
        thumb_megapixels=thumb_mp,
        scene_detect=scene_detect if is_video else False,
        output_folder=str(output_dir),
    )
    
    gen = execute_extraction(event, progress_queue, cancel_event, logger, config, thumbnail_manager, model_registry=model_registry)
    result = _run_pipeline(gen, "Extraction" if is_video else "Ingestion")
    
    # Verify output
    frame_map_path = output_dir / "frame_map.json"
    if frame_map_path.exists():
        frame_map = json.loads(frame_map_path.read_text())
        click.echo(f"   📊 Processed {len(frame_map)} frames/images")
    
    click.secho(f"\n✅ {'Extraction' if is_video else 'Ingestion'} complete. Output: {output_dir}", fg="green", bold=True)


@cli.command()
@click.option("--session", "-s", required=True, type=click.Path(exists=True), help="Path to extraction output directory.")
@click.option("--source", "-v", required=True, type=click.Path(exists=True), help="Path to original video file or image folder.")
@click.option("--face-ref", "-f", type=click.Path(exists=True), help="Path to reference face image for tracking.")
@click.option("--strategy", default="🧑‍🤝‍🧑 Find Prominent Person", help="Primary seed strategy.")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
@click.option("--resume", is_flag=True, help="Resume from last checkpoint.")
@click.option("--force", is_flag=True, help="Force re-run of all steps.")
def analyze(session, source, face_ref, strategy, verbose, resume, force):
    """
    Run full analysis pipeline on extracted frames or ingested photos.
    
    This runs pre-analysis (seed detection), mask propagation (video only), and metric analysis.
    Requires extraction/ingestion to have been run first.
    """
    output_dir = Path(session)
    source_path = Path(source)
    is_video = not source_path.is_dir()
    
    click.secho(f"\n🔬 ANALYSIS", fg="cyan", bold=True)
    click.echo(f"   Session: {output_dir}")
    click.echo(f"   Source: {source}")
    if face_ref:
        click.echo(f"   Face Ref: {face_ref}")
    
    config, logger, progress_queue, cancel_event, model_registry, thumbnail_manager = _setup_runtime(output_dir, verbose)
    
    # Build PreAnalysisEvent
    pre_event = PreAnalysisEvent(
        output_folder=str(output_dir),
        video_path=str(source) if is_video else "",
        resume=resume,
        enable_face_filter=bool(face_ref),
        face_ref_img_path=str(face_ref) if face_ref else "",
        face_model_name="buffalo_l",
        enable_subject_mask=True,
        tracker_model_name="sam3",
        best_frame_strategy="Largest Person",
        scene_detect=True if is_video else False,
        min_mask_area_pct=1.0,
        sharpness_base_scale=2500.0,
        edge_strength_base_scale=100.0,
        pre_analysis_enabled=True,
        pre_sample_nth=1,
        primary_seed_strategy=strategy,
        compute_quality_score=True,
        compute_sharpness=True,
        compute_edge_strength=True,
        compute_contrast=True,
        compute_brightness=True,
        compute_entropy=True,
        compute_eyes_open=True,
        compute_yaw=True,
        compute_pitch=True,
        compute_face_sim=True,
        compute_subject_mask_area=True,
        compute_niqe=True,
        compute_phash=True,
    )
    
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        click.echo(f"   🚀 CUDA available, using GPU acceleration")
        torch.set_float32_matmul_precision("medium")
    
    # Stage 1: Pre-Analysis
    click.secho("\n📍 Stage 1: Pre-Analysis (Seed Detection)", fg="yellow")
    gen = execute_pre_analysis(pre_event, progress_queue, cancel_event, logger, config, thumbnail_manager, cuda_available, progress=None, model_registry=model_registry)
    pre_result = _run_pipeline(gen, "Pre-Analysis")
    
    scenes = pre_result.get("scenes", [])
    click.echo(f"   📊 Found {len(scenes)} scenes")
    
    # Stage 2: Propagation (Skip for folders as they have no temporal tracking)
    prop_event = PropagationEvent(
        output_folder=str(output_dir),
        video_path=str(source) if is_video else "",
        scenes=scenes,
        analysis_params=pre_event,
    )

    if is_video:
        click.secho("\n📍 Stage 2: Mask Propagation", fg="yellow")
        gen = execute_propagation(prop_event, progress_queue, cancel_event, logger, config, thumbnail_manager, cuda_available, progress=None, model_registry=model_registry)
        _run_pipeline(gen, "Propagation")
    else:
        click.secho("\n📍 Stage 2: Mask Propagation (Skipped for Folder)", fg="yellow")
    
    # Stage 3: Analysis
    click.secho("\n📍 Stage 3: Metric Analysis", fg="yellow")
    gen = execute_analysis(prop_event, progress_queue, cancel_event, logger, config, thumbnail_manager, cuda_available, progress=None, model_registry=model_registry)
    _run_pipeline(gen, "Analysis")
    
    # Verify outputs
    db_path = output_dir / "metadata.db"
    mask_dir = output_dir / "masks"
    
    if db_path.exists():
        click.echo(f"   ✓ Metadata database: {db_path}")
    if mask_dir.exists() and any(mask_dir.iterdir()):
        click.echo(f"   ✓ Masks generated: {mask_dir}")
    
    click.secho(f"\n✅ Analysis complete. Results: {output_dir}", fg="green", bold=True)


@cli.command()
@click.option("--source", "-v", required=True, type=click.Path(exists=True), help="Path to input video file or image folder.")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output directory.")
@click.option("--face-ref", "-f", type=click.Path(exists=True), help="Path to reference face image.")
@click.option("--nth-frame", "-n", default=3, type=int, help="Extract every Nth frame (video only).")
@click.option("--max-resolution", "-r", default="1080", help="Max video resolution (video only).")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
@click.option("--clean", is_flag=True, help="Remove existing output directory.")
@click.option("--resume", is_flag=True, help="Resume from last checkpoint.")
@click.option("--force", is_flag=True, help="Force re-run of all steps.")
def full(source, output, face_ref, nth_frame, max_resolution, verbose, clean, resume, force):
    """
    Run the complete pipeline: extraction/ingestion + analysis.
    
    This is equivalent to running 'extract' followed by 'analyze'.
    """
    output_dir = Path(output)
    source_path = Path(source)
    is_video = not source_path.is_dir()
    
    if clean and output_dir.exists():
        click.echo(f"🧹 Cleaning existing output: {output_dir}")
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    click.secho(f"\n🚀 FULL PIPELINE", fg="cyan", bold=True)
    click.echo(f"   Source: {source}")
    click.echo(f"   Output: {output_dir}")
    
    config, logger, progress_queue, cancel_event, model_registry, thumbnail_manager = _setup_runtime(output_dir, verbose)
    cuda_available = torch.cuda.is_available()
    
    # --- EXTRACTION / INGESTION ---
    click.secho(f"\n━━━ STAGE: {'EXTRACTION' if is_video else 'INGESTION'} ━━━", fg="cyan", bold=True)
    ext_event = ExtractionEvent(
        source_path=str(source),
        method="every_nth_frame",
        interval="1.0",
        nth_frame=nth_frame,
        max_resolution=max_resolution,
        thumbnails_only=True,
        thumb_megapixels=0.5,
        scene_detect=True if is_video else False,
        output_folder=str(output_dir),
    )
    gen = execute_extraction(ext_event, progress_queue, cancel_event, logger, config, thumbnail_manager, model_registry=model_registry)
    _run_pipeline(gen, "Extraction" if is_video else "Ingestion")
    
    # --- PRE-ANALYSIS ---
    click.secho("\n━━━ STAGE: PRE-ANALYSIS ━━━", fg="cyan", bold=True)
    pre_event = PreAnalysisEvent(
        output_folder=str(output_dir),
        video_path=str(source) if is_video else "",
        resume=resume,
        enable_face_filter=bool(face_ref),
        face_ref_img_path=str(face_ref) if face_ref else "",
        face_model_name="buffalo_l",
        enable_subject_mask=True,
        tracker_model_name="sam3",
        best_frame_strategy="Largest Person",
        scene_detect=True if is_video else False,
        min_mask_area_pct=1.0,
        sharpness_base_scale=2500.0,
        edge_strength_base_scale=100.0,
        pre_analysis_enabled=True,
        pre_sample_nth=1,
        primary_seed_strategy="🧑‍🤝‍🧑 Find Prominent Person",
        compute_quality_score=True,
        compute_sharpness=True,
        compute_edge_strength=True,
        compute_contrast=True,
        compute_brightness=True,
        compute_entropy=True,
        compute_eyes_open=True,
        compute_yaw=True,
        compute_pitch=True,
        compute_face_sim=True,
        compute_subject_mask_area=True,
        compute_niqe=True,
        compute_phash=True,
    )
    gen = execute_pre_analysis(pre_event, progress_queue, cancel_event, logger, config, thumbnail_manager, cuda_available, progress=None, model_registry=model_registry)
    pre_result = _run_pipeline(gen, "Pre-Analysis")
    scenes = pre_result.get("scenes", [])
    click.echo(f"   📊 Found {len(scenes)} scenes")
    
    # --- PROPAGATION ---
    if is_video:
        click.secho("\n━━━ STAGE: PROPAGATION ━━━", fg="cyan", bold=True)
        prop_event = PropagationEvent(
            output_folder=str(output_dir),
            video_path=str(source),
            scenes=scenes,
            analysis_params=pre_event,
        )
        gen = execute_propagation(prop_event, progress_queue, cancel_event, logger, config, thumbnail_manager, cuda_available, progress=None, model_registry=model_registry)
        _run_pipeline(gen, "Propagation")
    else:
        click.secho("\n━━━ STAGE: PROPAGATION (Skipped for Folder) ━━━", fg="cyan", bold=True)
        prop_event = PropagationEvent(
            output_folder=str(output_dir),
            video_path="",
            scenes=scenes,
            analysis_params=pre_event,
        )
    
    # --- ANALYSIS ---
    click.secho("\n━━━ STAGE: ANALYSIS ━━━", fg="cyan", bold=True)
    gen = execute_analysis(prop_event, progress_queue, cancel_event, logger, config, thumbnail_manager, cuda_available, progress=None, model_registry=model_registry)
    _run_pipeline(gen, "Analysis")
    
    click.secho(f"\n🎉 PIPELINE COMPLETE", fg="green", bold=True)
    click.echo(f"   Results: {output_dir}")
    click.echo(f"   Metadata: {output_dir / 'metadata.db'}")
    if is_video:
        click.echo(f"   Masks: {output_dir / 'masks'}")


@cli.command()
@click.option("--session", "-s", required=True, type=click.Path(exists=True), help="Path to analysis output directory.")
def status(session):
    """
    Show the status of a processing session.
    
    Displays what stages have been completed and what data is available.
    """
    output_dir = Path(session)
    
    click.secho(f"\n📋 SESSION STATUS: {output_dir}", fg="cyan", bold=True)
    
    # Check for fingerprint
    from core.fingerprint import load_fingerprint
    fp = load_fingerprint(str(output_dir))
    if fp:
        click.secho(f"   ✓ Fingerprint: {fp.created_at}", fg="green")
        click.echo(f"      Video: {Path(fp.video_path).name}")
        click.echo(f"      Size: {fp.video_size:,} bytes")
    
    # Check for keys files
    checks = [
        ("frame_map.json", "Extraction complete"),
        ("scene_seeds.json", "Pre-analysis complete"),
        ("masks", "Propagation complete (directory exists)"),
        ("metadata.db", "Analysis complete"),
    ]
    
    for filename, description in checks:
        path = output_dir / filename
        exists = path.exists()
        status_icon = "✓" if exists else "✗"
        color = "green" if exists else "red"
        click.secho(f"   {status_icon} {description}", fg=color)
        if exists:
            if path.is_file():
                size = path.stat().st_size
                click.echo(f"      {filename} ({size:,} bytes)")
            else:
                count = len(list(path.iterdir())) if path.is_dir() else 0
                click.echo(f"      {filename}/ ({count} items)")
    
    # Check resumability
    progress_path = output_dir / "progress.json"
    if progress_path.exists():
        try:
            with open(progress_path) as f:
                prog = json.load(f)
            completed = len(prog.get("completed_scenes", []))
            click.secho(f"   🔄 Resumable: Yes ({completed} scenes completed)", fg="blue")
        except:
            pass


@cli.command()
@click.option("--session", "-s", required=True, type=click.Path(exists=True), help="Path to analysis output directory.")
@click.option("--quality-min", type=float, help="Minimum quality score (0-100).")
@click.option("--face-min", type=float, help="Minimum face similarity/match score (0-1).")
@click.option("--dedup/--no-dedup", default=True, help="Enable deduplication.")
@click.option("--dedup-method", type=click.Choice(["pHash", "SSIM", "LPIPS"]), default="pHash", help="Deduplication method.")
@click.option("--dedup-thresh", type=float, help="Deduplication threshold.")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
def filter(session, quality_min, face_min, dedup, dedup_method, dedup_thresh, verbose):
    """
    Filter analyzed frames based on quality and similarity.

    Loads metadata from the session database, applies specified thresholds,
    and reports the number of kept vs rejected frames.
    """
    output_dir = Path(session)
    db_path = output_dir / "metadata.db"
    
    if not db_path.exists():
        click.secho(f"❌ No metadata database found in {session}. Run 'analyze' first.", fg="red")
        sys.exit(1)
        
    click.secho(f"\n🔍 FILTERING: {output_dir}", fg="cyan", bold=True)
    
    config, logger, progress_queue, cancel_event, model_registry, thumbnail_manager = _setup_runtime(output_dir, verbose)
    
    # Load all frames from DB
    db = Database(db_path)
    all_frames = db.load_all_metadata()
    db.close()
    
    if not all_frames:
        click.secho("⚠️ No frames found in database.", fg="yellow")
        return

    # Build filter dict
    filters = {
        "quality_score_min": quality_min if quality_min is not None else config.filter_default_quality_score["default_min"],
        "face_sim_min": face_min if face_min is not None else config.filter_default_face_sim["default_min"],
        "enable_dedup": dedup,
        "dedup_method": dedup_method,
        "dedup_thresh": dedup_thresh if dedup_thresh is not None else (5 if dedup_method == "pHash" else 0.95),
    }
    
    click.echo(f"   📊 Applying filters to {len(all_frames)} frames...")
    if quality_min is not None:
        click.echo(f"      - Quality Min: {quality_min}")
    if face_min is not None:
        click.echo(f"      - Face Min: {face_min}")
    if dedup:
        click.echo(f"      - Dedup: {dedup_method} (thresh: {filters['dedup_thresh']})")

    kept, rejected, rejection_counts, reasons = apply_all_filters_vectorized(
        all_frames, 
        filters, 
        config, 
        thumbnail_manager=thumbnail_manager, 
        output_dir=str(output_dir)
    )
    
    click.secho(f"\n✅ Filtering complete:", fg="green", bold=True)
    click.echo(f"   🟢 Kept:     {len(kept)}")
    click.echo(f"   🔴 Rejected: {len(rejected)}")
    
    if rejection_counts:
        click.echo("\n   Rejection Reasons:")
        for reason, count in rejection_counts.most_common():
            click.echo(f"      - {reason}: {count}")


if __name__ == "__main__":
    cli()
