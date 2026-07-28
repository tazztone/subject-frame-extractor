import threading
from queue import Queue
from unittest.mock import MagicMock

from core.context import AnalysisContext


def test_analysis_context_initialization_without_progress():
    config_mock = MagicMock()
    logger_mock = MagicMock()
    progress_queue = Queue()
    cancel_event = threading.Event()
    thumbnail_manager_mock = MagicMock()
    model_registry_mock = MagicMock()
    cuda_available = True

    context = AnalysisContext(
        config=config_mock,
        logger=logger_mock,
        progress_queue=progress_queue,
        cancel_event=cancel_event,
        thumbnail_manager=thumbnail_manager_mock,
        model_registry=model_registry_mock,
        cuda_available=cuda_available,
    )

    assert context.config is config_mock
    assert context.logger is logger_mock
    assert context.progress_queue is progress_queue
    assert context.cancel_event is cancel_event
    assert context.thumbnail_manager is thumbnail_manager_mock
    assert context.model_registry is model_registry_mock
    assert context.cuda_available is cuda_available
    assert context.progress is None
    assert context.loaded_models == {}


def test_analysis_context_initialization_with_progress():
    config_mock = MagicMock()
    logger_mock = MagicMock()
    progress_queue = Queue()
    cancel_event = threading.Event()
    thumbnail_manager_mock = MagicMock()
    model_registry_mock = MagicMock()
    cuda_available = False
    progress_mock = MagicMock()

    context = AnalysisContext(
        config=config_mock,
        logger=logger_mock,
        progress_queue=progress_queue,
        cancel_event=cancel_event,
        thumbnail_manager=thumbnail_manager_mock,
        model_registry=model_registry_mock,
        cuda_available=cuda_available,
        progress=progress_mock,
    )

    assert context.config is config_mock
    assert context.logger is logger_mock
    assert context.progress_queue is progress_queue
    assert context.cancel_event is cancel_event
    assert context.thumbnail_manager is thumbnail_manager_mock
    assert context.model_registry is model_registry_mock
    assert context.cuda_available is cuda_available
    assert context.progress is progress_mock
    assert context.loaded_models == {}
