"""
CLI argument parsing and command definitions.
"""

import click

from core.cli_commands import (
    run_analyze,
    run_extract,
    run_filter,
    run_full,
    run_status,
)


@click.group()
def cli():
    """Subject Frame Extractor CLI."""
    pass


@cli.command()
@click.argument("source", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option(
    "--method",
    "-m",
    default="every_nth_frame",
    type=click.Choice(["every_nth_frame", "scene_changes"]),
    help="Extraction method.",
)
@click.option(
    "--nth-frame",
    "-n",
    default=3,
    type=click.IntRange(min=1),
    help="Extract every Nth frame (for every_nth_frame method).",
)
@click.option(
    "--max-resolution",
    "-r",
    default="1080",
    type=click.Choice(["480", "720", "1080", "1440", "2160"]),
    help="Max video resolution.",
)
@click.option("--thumb-mp", default=0.5, type=float, help="Thumbnail megapixels.")
@click.option("--scene-detect/--no-scene-detect", default=True, help="Enable scene detection (ignored for folders).")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
@click.option("--clean", is_flag=True, help="Remove existing output directory before extraction.")
@click.option("--force", is_flag=True, help="Force extraction even if fingerprint matches.")
def extract(source, output, method, nth_frame, max_resolution, thumb_mp, scene_detect, verbose, clean, force):
    """Extract frames from a video file or ingest an image folder."""
    run_extract(source, output, method, nth_frame, max_resolution, thumb_mp, scene_detect, verbose, clean, force)


@cli.command()
@click.option(
    "--session", "-s", required=True, type=click.Path(exists=True), help="Path to extraction output directory."
)
@click.option(
    "--source", "-v", required=True, type=click.Path(exists=True), help="Path to original video file or image folder."
)
@click.option("--face-ref", "-f", type=click.Path(exists=True), help="Path to reference face image for tracking.")
@click.option("--strategy", default="🧑‍🤝‍🧑 Find Prominent Person", help="Primary seed strategy.")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
@click.option("--resume", is_flag=True, help="Resume from last checkpoint.")
@click.option("--force", is_flag=True, help="Force re-run of all steps.")
def analyze(session, source, face_ref, strategy, verbose, resume, force):
    """Run full analysis pipeline on extracted frames or ingested photos."""
    run_analyze(session, source, face_ref, strategy, verbose, resume, force)


@cli.command()
@click.option(
    "--source", "-v", required=True, type=click.Path(exists=True), help="Path to input video file or image folder."
)
@click.option("--output", "-o", required=True, type=click.Path(), help="Output directory.")
@click.option("--face-ref", "-f", type=click.Path(exists=True), help="Path to reference face image.")
@click.option("--nth-frame", "-n", default=3, type=int, help="Extract every Nth frame (video only).")
@click.option("--max-resolution", "-r", default="1080", help="Max video resolution (video only).")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
@click.option("--clean", is_flag=True, help="Remove existing output directory.")
@click.option("--resume", is_flag=True, help="Resume from last checkpoint.")
@click.option("--force", is_flag=True, help="Force re-run of all steps.")
def full(source, output, face_ref, nth_frame, max_resolution, verbose, clean, resume, force):
    """Run the complete pipeline: extraction/ingestion + analysis."""
    run_full(source, output, face_ref, nth_frame, max_resolution, verbose, clean, resume, force)


@cli.command()
@click.option("--session", "-s", required=True, type=click.Path(exists=True), help="Path to analysis output directory.")
def status(session):
    """Show the status of a processing session."""
    run_status(session)


@cli.command()
@click.option("--session", "-s", required=True, type=click.Path(exists=True), help="Path to analysis output directory.")
@click.option("--quality-min", type=float, help="Minimum quality score (0-100).")
@click.option("--face-min", type=float, help="Minimum face similarity/match score (0-1).")
@click.option("--dedup/--no-dedup", default=True, help="Enable deduplication.")
@click.option(
    "--dedup-method", type=click.Choice(["pHash", "SSIM", "LPIPS"]), default="pHash", help="Deduplication method."
)
@click.option("--dedup-thresh", type=float, help="Deduplication threshold.")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
def filter(session, quality_min, face_min, dedup, dedup_method, dedup_thresh, verbose):
    """Filter analyzed frames based on quality and similarity."""
    run_filter(session, quality_min, face_min, dedup, dedup_method, dedup_thresh, verbose)
