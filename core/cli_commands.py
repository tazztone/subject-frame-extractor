"""
Core implementation of CLI commands.
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Optional

import click

from core.cli_utils import _run_pipeline, _setup_runtime
from core.database import Database
from core.events import ExtractionEvent, PreAnalysisEvent
from core.filtering import apply_all_filters_vectorized
from core.fingerprint import create_fingerprint, fingerprints_match, load_fingerprint
from core.pipelines import (
    execute_analysis_orchestrator,
    execute_extraction,
    execute_full_pipeline,
)
from core.utils.device import is_cuda_available


def run_extract(source, output, method, nth_frame, max_resolution, thumb_mp, scene_detect, verbose, clean, force):
    output_dir = Path(output)
    source_path = Path(source)
    is_video = not source_path.is_dir()

    if clean and output_dir.exists():
        click.echo(f"🧹 Cleaning existing output: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if is_video:
        ext_settings = {
            "method": method,
            "nth_frame": nth_frame,
            "max_resolution": str(max_resolution),
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

    config, logger, progress_queue, cancel_event, model_registry, thumbnail_manager = _setup_runtime(
        output_dir, verbose
    )

    event = ExtractionEvent(
        source_path=str(source),
        method=method,
        interval=1.0,
        nth_frame=nth_frame,
        max_resolution=str(max_resolution),
        thumbnails_only=True,
        thumb_megapixels=thumb_mp,
        scene_detect=scene_detect if is_video else False,
        output_folder=str(output_dir),
    )

    gen = execute_extraction(
        event, progress_queue, cancel_event, logger, config, thumbnail_manager, model_registry=model_registry
    )
    _run_pipeline(gen, "Extraction" if is_video else "Ingestion")

    frame_map_path = output_dir / "frame_map.json"
    if frame_map_path.exists():
        frame_map = json.loads(frame_map_path.read_text())
        click.echo(f"   📊 Processed {len(frame_map)} frames/images")

    click.secho(
        f"\n✅ {'Extraction' if is_video else 'Ingestion'} complete. Output: {output_dir}", fg="green", bold=True
    )


def run_analyze(session, source, face_ref, strategy, verbose, resume, force):
    output_dir = Path(session)
    source_path = Path(source)
    is_video = not source_path.is_dir()

    click.secho("\n🔬 ANALYSIS", fg="cyan", bold=True)
    click.echo(f"   Session: {output_dir}")
    click.echo(f"   Source: {source}")
    if face_ref:
        click.echo(f"   Face Ref: {face_ref}")

    config, logger, progress_queue, cancel_event, model_registry, thumbnail_manager = _setup_runtime(
        output_dir, verbose
    )

    pre_event = _build_pre_analysis_event(output_dir, source, is_video, face_ref, strategy, resume)

    cuda_available = is_cuda_available()
    if cuda_available:
        import torch

        click.echo("   🚀 CUDA available, using GPU acceleration")
        torch.set_float32_matmul_precision("medium")

    gen = execute_analysis_orchestrator(
        pre_event,
        progress_queue,
        cancel_event,
        logger,
        config,
        thumbnail_manager,
        cuda_available,
        progress=None,
        model_registry=model_registry,
    )
    _run_pipeline(gen, "Analysis Workflow")

    db_path = output_dir / "metadata.db"
    mask_dir = output_dir / "masks"

    if db_path.exists():
        click.echo(f"   ✓ Metadata database: {db_path}")
    if mask_dir.exists() and any(mask_dir.iterdir()):
        click.echo(f"   ✓ Masks generated: {mask_dir}")

    click.secho(f"\n✅ Analysis complete. Results: {output_dir}", fg="green", bold=True)


def run_status(session):
    output_dir = Path(session)
    click.secho(f"\n📋 SESSION STATUS: {output_dir}", fg="cyan", bold=True)

    fp = load_fingerprint(str(output_dir))
    if fp:
        click.secho(f"   ✓ Fingerprint: {fp.created_at}", fg="green")
        click.echo(f"      Video: {Path(fp.video_path).name}")
        click.echo(f"      Size: {fp.video_size:,} bytes")

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

    progress_path = output_dir / "progress.json"
    if progress_path.exists():
        try:
            with open(progress_path) as f:
                prog = json.load(f)
            completed = len(prog.get("completed_scenes", []))
            click.secho(f"   🔄 Resumable: Yes ({completed} scenes completed)", fg="blue")
        except Exception:
            pass


def run_filter(session, quality_min, face_min, dedup, dedup_method, dedup_thresh, verbose):
    output_dir = Path(session)
    db_path = output_dir / "metadata.db"

    if not db_path.exists():
        click.secho(f"❌ No metadata database found in {session}. Run 'analyze' first.", fg="red")
        sys.exit(1)

    click.secho(f"\n🔍 FILTERING: {output_dir}", fg="cyan", bold=True)

    config, logger, progress_queue, cancel_event, model_registry, thumbnail_manager = _setup_runtime(
        output_dir, verbose
    )

    db = Database(db_path)
    all_frames = db.load_all_metadata()
    db.close()

    if not all_frames:
        click.secho("⚠️ No frames found in database.", fg="yellow")
        return

    filters = {
        "quality_score_min": quality_min
        if quality_min is not None
        else config.filter_default_quality_score["default_min"],
        "face_sim_min": face_min if face_min is not None else config.filter_default_face_sim["default_min"],
        "enable_dedup": dedup,
        "dedup_method": dedup_method,
        "dedup_thresh": dedup_thresh if dedup_thresh is not None else (5 if dedup_method == "pHash" else 0.95),
    }

    click.echo(f"   📊 Applying filters to {len(all_frames)} frames...")
    kept, rejected, rejection_counts, reasons = apply_all_filters_vectorized(
        all_frames, filters, config, thumbnail_manager=thumbnail_manager, output_dir=str(output_dir)
    )

    click.secho("\n✅ Filtering complete:", fg="green", bold=True)
    click.echo(f"   🟢 Kept:     {len(kept)}")
    click.echo(f"   🔴 Rejected: {len(rejected)}")

    if rejection_counts:
        click.echo("\n   Rejection Reasons:")
        for reason, count in rejection_counts.most_common():
            click.echo(f"      - {reason}: {count}")


def run_full(source, output, face_ref, nth_frame, max_resolution, verbose, clean, resume, force):
    output_dir = Path(output)
    source_path = Path(source)
    is_video = not source_path.is_dir()

    if clean and output_dir.exists():
        click.echo(f"🧹 Cleaning existing output: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    click.secho("\n🚀 FULL PIPELINE", fg="cyan", bold=True)
    click.echo(f"   Source: {source}")
    click.echo(f"   Output: {output_dir}")

    config, logger, progress_queue, cancel_event, model_registry, thumbnail_manager = _setup_runtime(
        output_dir, verbose
    )
    cuda_available = is_cuda_available()

    ext_event = ExtractionEvent(
        source_path=str(source),
        method="every_nth_frame",
        interval=1.0,
        nth_frame=nth_frame,
        max_resolution=str(max_resolution),
        thumbnails_only=True,
        thumb_megapixels=0.5,
        scene_detect=True if is_video else False,
        output_folder=str(output_dir),
    )
    gen = execute_full_pipeline(
        ext_event,
        progress_queue,
        cancel_event,
        logger,
        config,
        thumbnail_manager,
        cuda_available,
        progress=None,
        model_registry=model_registry,
    )
    _run_pipeline(gen, "Full Pipeline")

    click.secho("\n🎉 PIPELINE COMPLETE", fg="green", bold=True)
    click.echo(f"   Results: {output_dir}")
    click.echo(f"   Metadata: {output_dir / 'metadata.db'}")
    if is_video:
        click.echo(f"   Masks: {output_dir / 'masks'}")


def _build_pre_analysis_event(
    output_dir: Path,
    source: str,
    is_video: bool,
    face_ref: Optional[str],
    strategy: str,
    resume: bool,
) -> PreAnalysisEvent:
    """Helper to build consistent PreAnalysisEvent for CLI commands."""
    return PreAnalysisEvent(
        output_folder=str(output_dir),
        video_path=str(source) if is_video else "",
        resume=resume,
        face_ref_img_path=str(face_ref) if face_ref else "",
        face_model_name="buffalo_l",
        enable_subject_mask=True,
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
        compute_face_sim=bool(face_ref),
        compute_subject_mask_area=True,
        compute_niqe=True,
        compute_phash=True,
    )
