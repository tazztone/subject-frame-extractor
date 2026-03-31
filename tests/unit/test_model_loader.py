import hashlib
import io
from unittest.mock import MagicMock, patch

import pytest

# Mock dependencies to avoid hardware/import issues
with patch.dict(
    "sys.modules",
    {
        "lpips": MagicMock(),
        "torch": MagicMock(),
        "insightface": MagicMock(),
        "cv2": MagicMock(),
    },
):
    from core.io_utils import _compute_sha256, download_model
    from core.managers.model_loader import get_lpips_metric, initialize_analysis_models


class TestModelLoader:
    """
    Tests for core/managers/model_loader.py and core/io_utils.py (download_model)
    """

    @pytest.fixture
    def mock_deps(self):
        logger = MagicMock()
        error_handler = MagicMock()
        # Mock retry decorator to just call the function
        error_handler.with_retry.side_effect = lambda **kwargs: lambda f: f
        return logger, error_handler

    def test_get_lpips_metric(self):
        import lpips

        get_lpips_metric("alex", "cpu")
        lpips.LPIPS.assert_called_with(net="alex")

    def test_download_model_happy_path(self, mock_deps, tmp_path):
        logger, error_handler = mock_deps
        dest_path = tmp_path / "model.pt"
        content = b"fake model content"
        expected_sha = hashlib.sha256(content).hexdigest()

        # Mock urllib.request
        mock_resp = MagicMock()
        mock_resp.getheader.side_effect = lambda h: str(len(content)) if h == "Content-Length" else None
        mock_resp.read.side_effect = io.BytesIO(content).read
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp), patch("urllib.request.Request") as mock_req:
            download_model(
                url="http://fake.url/model.pt",
                dest_path=dest_path,
                description="Test Model",
                logger=logger,
                error_handler=error_handler,
                user_agent="test-agent",
                expected_sha256=expected_sha,
                min_size=10,  # ensure content is larger than min_size
            )

            assert dest_path.exists()
            assert dest_path.read_bytes() == content
            mock_req.assert_called_once_with("http://fake.url/model.pt", headers={"User-Agent": "test-agent"})
            logger.info.assert_any_call("Test Model downloaded and verified successfully.")

    def test_download_model_checksum_mismatch(self, mock_deps, tmp_path):
        logger, error_handler = mock_deps
        dest_path = tmp_path / "model.pt"
        content = b"fake model content"
        wrong_sha = "wrongsha256"

        # Mock urllib.request
        mock_resp = MagicMock()
        mock_resp.getheader.side_effect = lambda h: str(len(content)) if h == "Content-Length" else None
        mock_resp.read.side_effect = io.BytesIO(content).read
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            with pytest.raises(RuntimeError, match="Failed to download required model"):
                download_model(
                    url="http://fake.url/model.pt",
                    dest_path=dest_path,
                    description="Test Model",
                    logger=logger,
                    error_handler=error_handler,
                    user_agent="test-agent",
                    expected_sha256=wrong_sha,
                    min_size=5,
                )

            # Check that failed file is unlinked
            assert not dest_path.exists()

    def test_download_model_cached_valid(self, mock_deps, tmp_path):
        logger, error_handler = mock_deps
        dest_path = tmp_path / "model.pt"
        content = b"cached content"
        dest_path.write_bytes(content)
        sha = hashlib.sha256(content).hexdigest()

        with patch("urllib.request.urlopen") as mock_url:
            download_model(
                url="http://fake.url/model.pt",
                dest_path=dest_path,
                description="Cached Model",
                logger=logger,
                error_handler=error_handler,
                user_agent="test",
                expected_sha256=sha,
            )
            mock_url.assert_not_called()
            logger.info.assert_any_call(f"Using cached and verified Cached Model: {dest_path}")

    def test_download_model_directory_creation(self, mock_deps, tmp_path):
        logger, error_handler = mock_deps
        nested_dest = tmp_path / "deeply" / "nested" / "model.pt"
        content = b"a" * 20

        mock_resp = MagicMock()
        mock_resp.getheader.side_effect = lambda h: str(len(content)) if h == "Content-Length" else None
        mock_resp.read.side_effect = io.BytesIO(content).read
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            download_model(
                url="http://fake.url/model.pt",
                dest_path=nested_dest,
                description="Nested Model",
                logger=logger,
                error_handler=error_handler,
                user_agent="test",
                min_size=10,
            )
            assert nested_dest.exists()

    def test_initialize_analysis_models_basic(self, tmp_path):
        from core.models import AnalysisParameters

        config = MagicMock()
        config.models_dir = tmp_path / "models"
        config.face_landmarker_url = "http://fake.url/landmarker.task"
        config.face_landmarker_sha256 = None

        params = AnalysisParameters(output_folder=str(tmp_path), video_path="test.mp4", compute_face_sim=False)
        logger = MagicMock()
        registry = MagicMock()

        with (
            patch("core.managers.model_loader.download_model"),
            patch("core.managers.model_loader.get_face_landmarker", return_value="mock_landmarker"),
        ):
            # Create dummy file to avoid download logic for landmarker check
            (tmp_path / "models").mkdir()
            (tmp_path / "models" / "landmarker.task").touch()

            res = initialize_analysis_models(params, config, logger, registry)
            assert res["face_landmarker"] == "mock_landmarker"
            assert res["face_analyzer"] is None

    def test_compute_sha256(self, tmp_path):
        p = tmp_path / "test.txt"
        p.write_bytes(b"hello")
        expected = hashlib.sha256(b"hello").hexdigest()
        assert _compute_sha256(p) == expected
