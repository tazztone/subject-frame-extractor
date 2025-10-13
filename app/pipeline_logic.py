import threading
from dataclasses import asdict
from queue import Queue

from app.config import Config
from app.events import (ExtractionEvent, PreAnalysisEvent, PropagationEvent,
                              SessionLoadEvent)
from app.logging import UnifiedLogger
from app.models import AnalysisParameters
from app.pipelines import (ExtractionPipeline, PreAnalysisPipeline,
                                 PropagationPipeline, SessionLoadPipeline)


def execute_extraction(event: ExtractionEvent, progress_queue: Queue,
                       cancel_event: threading.Event, logger: UnifiedLogger,
                       config: Config):
    """Execute extraction pipeline."""
    params = AnalysisParameters.from_ui(**asdict(event))
    pipeline = ExtractionPipeline(params, progress_queue, cancel_event, logger,
                                  config, None)
    yield pipeline.run()


def execute_pre_analysis(event: PreAnalysisEvent, progress_queue: Queue,
                         cancel_event: threading.Event, logger: UnifiedLogger,
                         config: Config, thumbnail_manager, cuda_available):
    """Execute pre-analysis pipeline."""
    params_dict = asdict(event)
    params_dict['cuda_available'] = cuda_available
    params = AnalysisParameters.from_ui(**params_dict)
    pipeline = PreAnalysisPipeline(params, progress_queue, cancel_event,
                                   logger, config, thumbnail_manager)
    yield from pipeline.run()


def execute_propagation(event: PropagationEvent, progress_queue: Queue,
                        cancel_event: threading.Event, logger: UnifiedLogger,
                        config: Config, thumbnail_manager, cuda_available):
    """Execute propagation pipeline."""
    params_dict = asdict(event)
    params_dict['cuda_available'] = cuda_available
    params = AnalysisParameters.from_ui(**params_dict)
    pipeline = PropagationPipeline(params, progress_queue, cancel_event,
                                   logger, config, thumbnail_manager)
    yield from pipeline.run()


def execute_session_load(event: SessionLoadEvent, logger: UnifiedLogger,
                         config: Config, thumbnail_manager):
    """Loads a session from a previous run and prepares the UI."""
    params = AnalysisParameters.from_ui(**asdict(event))
    pipeline = SessionLoadPipeline(params, None, None, logger, config,
                                   thumbnail_manager)
    yield from pipeline.run()