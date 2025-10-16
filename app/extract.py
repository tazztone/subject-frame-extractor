"""Video frame extraction pipeline."""

import logging
from pathlib import Path

from app.base import Pipeline
from app.error_handling import ErrorHandler


class ExtractionPipeline(Pipeline):
    """Pipeline for extracting frames from video sources."""
    def __init__(self, params, progress_queue, cancel_event, logger=None, tracker=None):
        super().__init__(params, progress_queue, cancel_event, logger, tracker)
        self.config = None # This will be set by the EnhancedExtractionPipeline

    def run(self):
        """Run the extraction pipeline."""
        from app.utils import sanitize_filename
        from app.video import VideoManager, run_scene_detection, run_ffmpeg_extraction

        config = self.config

        self.tracker.start_stage("Preparing Video", 1)
        self.logger.info("Preparing video source...")
        vid_manager = VideoManager(self.params.source_path,
                                    self.params.max_resolution)
        video_path = Path(vid_manager.prepare_video(config, self.logger))

        output_dir = config.DIRS['downloads'] / video_path.stem
        output_dir.mkdir(exist_ok=True)
        self.tracker.update_progress(stage_items_processed=1)

        self.logger.info("Video ready",
                            user_context={'path': sanitize_filename(video_path.name)})

        video_info = VideoManager.get_video_info(video_path)

        if self.params.scene_detect:
            self.tracker.start_stage("Scene Detection")
            self._run_scene_detection(video_path, output_dir)

        self.tracker.start_stage("FFmpeg Extraction")
        self._run_ffmpeg(video_path, output_dir, video_info)

        if self.cancel_event.is_set():
            self.logger.info("Extraction cancelled by user.")
            return

        self.logger.success("Extraction complete.")
        return {
            "done": True,
            "output_dir": str(output_dir),
            "video_path": str(video_path)
        }

    def _run_scene_detection(self, video_path, output_dir):
        """Run scene detection on the video."""
        from app.video import run_scene_detection
        return run_scene_detection(video_path, output_dir, self.logger)

    def _run_ffmpeg(self, video_path, output_dir, video_info):
        """Run FFmpeg extraction."""
        from app.video import run_ffmpeg_extraction
        return run_ffmpeg_extraction(video_path, output_dir, video_info,
                                    self.params, self.progress_queue,
                                    self.cancel_event, self.logger, self.tracker)


from app.config import Config

class EnhancedExtractionPipeline(ExtractionPipeline):
    def __init__(self, params, progress_queue, cancel_event, config: 'Config', logger=None, tracker=None):
        super().__init__(params, progress_queue, cancel_event, logger, tracker)
        self.config = config
        self.error_handler = ErrorHandler(self.logger, self.config)
        self.tracker = tracker
        self.run = self.error_handler.with_retry(max_attempts=3, backoff_seconds=[1, 5, 15])(self.run)