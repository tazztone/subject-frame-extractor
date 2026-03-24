import json
import logging
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.io_utils import atomic_write_text
from core.logger import JSONFormatter, setup_logging
from core.managers.registry import ModelRegistry
from core.sam3_patches import apply_patches
from core.xmp_writer import write_xmp_sidecar


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


class TestRobustnessPhase2:
    # --- 1. Structured Logging Tests ---

    def test_json_formatter(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py", lineno=10, msg="Hello World", args=None, exc_info=None
        )
        record.component = "test_comp"
        record.custom_attr = "val"

        json_output = formatter.format(record)
        data = json.loads(json_output)

        assert data["message"] == "Hello World"
        assert data["component"] == "test_comp"
        assert data["custom_fields"]["custom_attr"] == "val"
        assert "timestamp" in data

    def test_setup_logging_structured(self, temp_dir):
        config = MagicMock()
        config.logs_dir = str(temp_dir)
        config.log_format = "%(message)s"
        config.log_colored = False
        config.log_level = "INFO"
        config.log_structured_path = "structured.jsonl"

        setup_logging(config)

        logger = logging.getLogger("app_logger")
        logger.info("Structured log test", extra={"component": "test_json"})

        structured_file = temp_dir / "structured.jsonl"
        assert structured_file.exists()

        with open(structured_file, "r") as f:
            lines = f.readlines()
            # Find our specific log line (might be others from setup)
            found = False
            for line in lines:
                data = json.loads(line)
                if data["message"] == "Structured log test":
                    assert data["component"] == "test_json"
                    found = True
                    break
            assert found

    # --- 2. Atomic Write Tests ---

    def test_atomic_write_text(self, temp_dir):
        target = temp_dir / "test.txt"
        content = "atomic content"

        atomic_write_text(target, content)

        assert target.exists()
        assert target.read_text() == content

    def test_xmp_writer_atomic(self, temp_dir):
        source = temp_dir / "image.jpg"
        source.touch()

        with patch("os.replace") as mock_replace:
            # We don't want to actually replace for this test, but check it's called
            write_xmp_sidecar(source, 5, "Green")
            assert mock_replace.called
            # The first arg is the temp file, second is the .xmp
            assert str(mock_replace.call_args[0][1]).endswith("image.xmp")

    # --- 3. SAM3 Patch Safety Tests ---

    @patch("core.sam3_patches.hashlib.sha256")
    @patch("core.sam3_patches.logger")
    def test_sam3_patch_hash_mismatch(self, mock_logger, mock_hash_cls, temp_dir):
        # Mocking import of sam3
        mock_svp = MagicMock()
        mock_svp.__file__ = str(temp_dir / "fake_predictor.py")
        Path(mock_svp.__file__).touch()

        mock_hash_obj = MagicMock()
        mock_hash_obj.hexdigest.return_value = "WRONG_HASH"
        mock_hash_cls.return_value = mock_hash_obj

        # Isolate from existing sys.modules
        module_name = "sam3.model.sam3_video_predictor"
        with patch.dict("sys.modules", {module_name: mock_svp}):
            # Force apply_patches to see the mock
            with patch("importlib.import_module", return_value=mock_svp):
                apply_patches()

        # Should warning about mismatch
        mock_logger.warning.assert_called()
        assert "SAM3 version mismatch" in mock_logger.warning.call_args[0][0]

    # --- 4. ML Model Sticky Failures ---

    def test_model_registry_sticky_failure(self):
        registry = ModelRegistry()
        loader = MagicMock(side_effect=RuntimeError("Hard Failure"))

        # First attempt fails
        with pytest.raises(RuntimeError):
            registry.get_or_load("bad_model", loader)

        assert "bad_model" in registry._failed_models
        assert loader.call_count == 1

        # Second attempt should NOT call loader and return None
        res = registry.get_or_load("bad_model", loader)
        assert res is None
        assert loader.call_count == 1
