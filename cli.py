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
from core.logger import AppLogger
from core.managers import ModelRegistry, ThumbnailManager
from core.pipelines import (
    execute_analysis,
    execute_extraction,
    execute_pre_analysis,
    execute_propagation,
)


def _setup_runtime(output_dir: Path, verbose: bool = False):
    """Initialize shared runtime components."""
    config = Config()
    config.monitoring_memory_critical_threshold_mb = 64000  # Disable watchdog for CLI
    config.log_level = "DEBUG" if verbose else "INFO"

    logger = AppLogger(config, log_dir=output_dir, log_to_file=True)
    progress_queue = Queue()
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
@click.option("--video", "-v", required=True, type=click.Path(exists=True), help="Path to input video file.")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output directory for extracted frames.")
@click.option("--method", "-m", default="every_nth_frame", type=click.Choice(["every_nth_frame", "scene", "keyframes"]), help="Extraction method.")
@click.option("--nth-frame", "-n", default=3, type=int, help="Extract every Nth frame (for every_nth_frame method).")
@click.option("--max-resolution", "-r", default="1080", type=click.Choice(["480", "720", "1080", "1440", "2160"]), help="Max video resolution.")
@click.option("--thumb-mp", default=0.5, type=float, help="Thumbnail megapixels.")
@click.option("--scene-detect/--no-scene-detect", default=True, help="Enable scene detection.")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
@click.option("--clean", is_flag=True, help="Remove existing output directory before extraction.")
def extract(video, output, method, nth_frame, max_resolution, thumb_mp, scene_detect, verbose, clean):
    """
    Extract frames from a video file.
    
    This is the first step in the pipeline. It extracts thumbnails from the video
    and optionally detects scene boundaries.
    """
    output_dir = Path(output)
    
    if clean and output_dir.exists():
        click.echo(f"🧹 Cleaning existing output: {output_dir}")
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    click.secho(f"\n🎬 EXTRACTION", fg="cyan", bold=True)
    click.echo(f"   Video: {video}")
    click.echo(f"   Output: {output_dir}")
    click.echo(f"   Method: {method} (every {nth_frame} frames)")
    
    config, logger, progress_queue, cancel_event, model_registry, thumbnail_manager = _setup_runtime(output_dir, verbose)
    
    event = ExtractionEvent(
        source_path=str(video),
        method=method,
        interval="1.0",
        nth_frame=nth_frame,
        max_resolution=max_resolution,
        thumbnails_only=True,
        thumb_megapixels=thumb_mp,
        scene_detect=scene_detect,
        output_folder=str(output_dir),
    )
    
    gen = execute_extraction(event, progress_queue, cancel_event, logger, config, thumbnail_manager, model_registry=model_registry)
    result = _run_pipeline(gen, "Extraction")
    
    # Verify output
    frame_map_path = output_dir / "frame_map.json"
    if frame_map_path.exists():
        frame_map = json.loads(frame_map_path.read_text())
        click.echo(f"   📊 Extracted {len(frame_map)} frames")
    
    click.secho(f"\n✅ Extraction complete. Output: {output_dir}", fg="green", bold=True)


@cli.command()
@click.option("--session", "-s", required=True, type=click.Path(exists=True), help="Path to extraction output directory.")
@click.option("--video", "-v", required=True, type=click.Path(exists=True), help="Path to original video file.")
@click.option("--face-ref", "-f", type=click.Path(exists=True), help="Path to reference face image for tracking.")
@click.option("--strategy", default="🧑‍🤝‍🧑 Find Prominent Person", help="Primary seed strategy.")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
def analyze(session, video, face_ref, strategy, verbose):
    """
    Run full analysis pipeline on extracted frames.
    
    This runs pre-analysis (seed detection), mask propagation, and metric analysis.
    Requires extraction to have been run first.
    """
    output_dir = Path(session)
    
    click.secho(f"\n🔬 ANALYSIS", fg="cyan", bold=True)
    click.echo(f"   Session: {output_dir}")
    click.echo(f"   Video: {video}")
    if face_ref:
        click.echo(f"   Face Ref: {face_ref}")
    
    config, logger, progress_queue, cancel_event, model_registry, thumbnail_manager = _setup_runtime(output_dir, verbose)
    
    # Build PreAnalysisEvent
    pre_event = PreAnalysisEvent(
        output_folder=str(output_dir),
        video_path=str(video),
        resume=False,
        enable_face_filter=bool(face_ref),
        face_ref_img_path=str(face_ref) if face_ref else "",
        face_model_name="buffalo_l",
        enable_subject_mask=True,
        tracker_model_name="sam3",
        best_frame_strategy="Largest Person",
        scene_detect=True,
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
    
    # Stage 2: Propagation
    click.secho("\n📍 Stage 2: Mask Propagation", fg="yellow")
    prop_event = PropagationEvent(
        output_folder=str(output_dir),
        video_path=str(video),
        scenes=scenes,
        analysis_params=pre_event,
    )
    gen = execute_propagation(prop_event, progress_queue, cancel_event, logger, config, thumbnail_manager, cuda_available, progress=None, model_registry=model_registry)
    _run_pipeline(gen, "Propagation")
    
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
@click.option("--video", "-v", required=True, type=click.Path(exists=True), help="Path to input video file.")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output directory.")
@click.option("--face-ref", "-f", type=click.Path(exists=True), help="Path to reference face image.")
@click.option("--nth-frame", "-n", default=3, type=int, help="Extract every Nth frame.")
@click.option("--max-resolution", "-r", default="1080", help="Max video resolution.")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
@click.option("--clean", is_flag=True, help="Remove existing output directory.")
def full(video, output, face_ref, nth_frame, max_resolution, verbose, clean):
    """
    Run the complete pipeline: extraction + analysis.
    
    This is equivalent to running 'extract' followed by 'analyze'.
    """
    output_dir = Path(output)
    
    if clean and output_dir.exists():
        click.echo(f"🧹 Cleaning existing output: {output_dir}")
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    click.secho(f"\n🚀 FULL PIPELINE", fg="cyan", bold=True)
    click.echo(f"   Video: {video}")
    click.echo(f"   Output: {output_dir}")
    
    config, logger, progress_queue, cancel_event, model_registry, thumbnail_manager = _setup_runtime(output_dir, verbose)
    cuda_available = torch.cuda.is_available()
    
    # --- EXTRACTION ---
    click.secho("\n━━━ STAGE: EXTRACTION ━━━", fg="cyan", bold=True)
    ext_event = ExtractionEvent(
        source_path=str(video),
        method="every_nth_frame",
        interval="1.0",
        nth_frame=nth_frame,
        max_resolution=max_resolution,
        thumbnails_only=True,
        thumb_megapixels=0.5,
        scene_detect=True,
        output_folder=str(output_dir),
    )
    gen = execute_extraction(ext_event, progress_queue, cancel_event, logger, config, thumbnail_manager, model_registry=model_registry)
    _run_pipeline(gen, "Extraction")
    
    # --- PRE-ANALYSIS ---
    click.secho("\n━━━ STAGE: PRE-ANALYSIS ━━━", fg="cyan", bold=True)
    pre_event = PreAnalysisEvent(
        output_folder=str(output_dir),
        video_path=str(video),
        resume=False,
        enable_face_filter=bool(face_ref),
        face_ref_img_path=str(face_ref) if face_ref else "",
        face_model_name="buffalo_l",
        enable_subject_mask=True,
        tracker_model_name="sam3",
        best_frame_strategy="Largest Person",
        scene_detect=True,
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
    click.secho("\n━━━ STAGE: PROPAGATION ━━━", fg="cyan", bold=True)
    prop_event = PropagationEvent(
        output_folder=str(output_dir),
        video_path=str(video),
        scenes=scenes,
        analysis_params=pre_event,
    )
    gen = execute_propagation(prop_event, progress_queue, cancel_event, logger, config, thumbnail_manager, cuda_available, progress=None, model_registry=model_registry)
    _run_pipeline(gen, "Propagation")
    
    # --- ANALYSIS ---
    click.secho("\n━━━ STAGE: ANALYSIS ━━━", fg="cyan", bold=True)
    gen = execute_analysis(prop_event, progress_queue, cancel_event, logger, config, thumbnail_manager, cuda_available, progress=None, model_registry=model_registry)
    _run_pipeline(gen, "Analysis")
    
    click.secho(f"\n🎉 PIPELINE COMPLETE", fg="green", bold=True)
    click.echo(f"   Results: {output_dir}")
    click.echo(f"   Metadata: {output_dir / 'metadata.db'}")
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
    
    # Check for key files
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


if __name__ == "__main__":
    cli()
