"""Video frame extraction pipeline."""

import logging
from pathlib import Path

from app.base import Pipeline
from app.error_handling import ErrorHandler


class ExtractionPipeline(Pipeline):
    """Pipeline for extracting frames from video sources."""

    def run(self):
        """Run the extraction pipeline."""
        from app.config import Config
        from app.logging import StructuredFormatter
        from app.utils import sanitize_filename
        from app.video import VideoManager, run_scene_detection, run_ffmpeg_extraction
        
        config = Config()
        run_log_handler = None
        
        try:
            self.logger.info("Preparing video source...")
            vid_manager = VideoManager(self.params.source_path, 
                                     self.params.max_resolution)
            video_path = Path(vid_manager.prepare_video())

            output_dir = config.DIRS['downloads'] / video_path.stem
            output_dir.mkdir(exist_ok=True)

            run_log_path = output_dir / "extraction_run.log"
            run_log_handler = logging.FileHandler(run_log_path, mode='w', 
                                                 encoding='utf-8')
            formatter = StructuredFormatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            run_log_handler.setFormatter(formatter)
            self.logger.add_handler(run_log_handler)
            
            self.logger.info("Video ready", 
                           extra={'path': sanitize_filename(video_path.name)})

            video_info = VideoManager.get_video_info(video_path)

            if self.params.scene_detect:
                self._run_scene_detection(video_path, output_dir)

            self._run_ffmpeg(video_path, output_dir, video_info)

            if self.cancel_event.is_set():
                return {"log": "Extraction cancelled."}
                
            self.logger.success("Extraction complete.")
            return {
                "done": True,
                "output_dir": str(output_dir),
                "video_path": str(video_path)
            }
        except Exception as e:
            return self.logger.pipeline_error("extraction", e)
        finally:
            self.logger.remove_handler(run_log_handler)

    def _run_scene_detection(self, video_path, output_dir):
        """Run scene detection on the video."""
        from app.video import run_scene_detection
        return run_scene_detection(video_path, output_dir)

    def _run_ffmpeg(self, video_path, output_dir, video_info):
        """Run FFmpeg extraction."""
        from app.video import run_ffmpeg_extraction
        return run_ffmpeg_extraction(video_path, output_dir, video_info,
                                    self.params, self.progress_queue,
                                    self.cancel_event)


from app.config import Config

class EnhancedExtractionPipeline(ExtractionPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = Config()
        self.error_handler = ErrorHandler(self.logger, self.config)
        self.run = self.error_handler.with_retry(max_attempts=3, backoff_seconds=[1, 5, 15])(self.run)
