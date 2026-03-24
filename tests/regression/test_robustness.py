from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from core.database import Database
from core.events import ExtractionEvent
from core.io_utils import download_model
from core.managers.registry import ModelRegistry
from core.managers.thumbnails import ThumbnailManager


@pytest.fixture
def mock_error_handler():
    """Mock ErrorHandler."""
    return MagicMock()


def test_registry_oom_retry(mock_logger):
    """Verify that ModelRegistry retries on OOM after clearing cache."""
    registry = ModelRegistry(logger=mock_logger)
    loader_mock = MagicMock()
    # First call raises OOM, second succeeds
    loader_mock.side_effect = [torch.cuda.OutOfMemoryError("CUDA out of memory"), "model_instance"]
    with patch.object(registry, "clear") as clear_spy:
        model = registry.get_or_load("test_model", loader_mock)
        assert model == "model_instance"
        assert loader_mock.call_count == 2
        clear_spy.assert_called_once()
        mock_logger.warning.assert_any_call("CUDA OOM loading 'test_model'. Clearing models and retrying...")


def test_database_atexit_registration(tmp_path):
    """Verify that Database registers an atexit handler."""
    db_path = tmp_path / "test.db"
    with patch("atexit.register") as atexit_spy:
        Database(db_path)
        atexit_spy.assert_called()


def test_thumbnail_memory_limit(mock_logger, mock_config, tmp_path):
    """Verify ThumbnailManager evicts based on byte size."""
    mock_config.thumbnail_cache_max_mb = 1  # 1MB limit
    tm = ThumbnailManager(logger=mock_logger, config=mock_config)
    # img1: 500*500*3 = 750,000 bytes (~0.7MB)
    img1 = np.zeros((500, 500, 3), dtype=np.uint8)
    # img2: 600*600*3 = 1,080,000 bytes (~1.0MB)
    img2 = np.zeros((600, 600, 3), dtype=np.uint8)
    p1 = tmp_path / "img1.png"
    p2 = tmp_path / "img2.png"
    p1.touch()
    p2.touch()
    with patch("PIL.Image.open") as mock_open:
        mock_img1 = MagicMock()
        mock_img1.convert.return_value = img1
        mock_img2 = MagicMock()
        mock_img2.convert.return_value = img2
        mock_open.side_effect = [MagicMock(__enter__=lambda s: mock_img1), MagicMock(__enter__=lambda s: mock_img2)]
        tm.get(p1)
        assert tm.current_bytes == img1.nbytes
        assert len(tm.cache) == 1
        tm.get(p2)
        assert tm.current_bytes == img2.nbytes
        assert len(tm.cache) == 1
        assert p1 not in tm.cache
        assert p2 in tm.cache


def test_download_content_length_abort(mock_logger, mock_error_handler):
    """Verify that download_model aborts early if Content-Length is too small."""
    url = "http://example.com/model.pt"
    dest = Path("/tmp/fake_model.pt")
    mock_resp = MagicMock()
    mock_resp.getheader.return_value = "500"  # 500 bytes
    mock_error_handler.with_retry.return_value = lambda f: f
    with patch("urllib.request.urlopen", return_value=MagicMock(__enter__=lambda s: mock_resp)):
        with pytest.raises(RuntimeError, match="Failed to download required model: model"):
            download_model(
                url=url,
                dest_path=dest,
                description="model",
                logger=mock_logger,
                error_handler=mock_error_handler,
                user_agent="test-agent",
                min_size=1000000,
            )


def test_extraction_event_validation():
    """Verify Pydantic validation for ExtractionEvent."""
    valid_payload = {
        "source_path": "test.mp4",
        "output_folder": "/tmp/out",
        "method": "all",
        "max_resolution": "1080p",
        "scene_detect": True,
    }
    with patch("core.events.validate_writable_directory", side_effect=lambda v, f: v):
        ExtractionEvent(**valid_payload)
    with patch("core.events.validate_writable_directory", side_effect=ValueError("Output Folder is not writable")):
        with pytest.raises(ValueError, match="Output Folder is not writable"):
            ExtractionEvent(**valid_payload)
    with pytest.raises(ValueError, match="nth_frame"):
        ExtractionEvent(**{**valid_payload, "nth_frame": 0})
