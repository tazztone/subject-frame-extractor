from unittest.mock import MagicMock, patch

import pytest

from core.cli_commands import run_analyze, run_extract, run_filter, run_full, run_status


class TestCLICommands:
    """
    Tests for core/cli_commands.py
    Mocks the underlying pipelines to test CLI orchestration and output.
    """

    @pytest.fixture
    def mock_runtime(self):
        config = MagicMock()
        logger = MagicMock()
        progress_queue = MagicMock()
        cancel_event = MagicMock()
        model_registry = MagicMock()
        thumbnail_manager = MagicMock()

        with patch(
            "core.cli_commands._setup_runtime",
            return_value=(config, logger, progress_queue, cancel_event, model_registry, thumbnail_manager),
        ):
            yield config, logger, progress_queue, cancel_event, model_registry, thumbnail_manager

    def test_run_extract_video(self, mock_runtime, tmp_path):
        source = tmp_path / "test.mp4"
        source.touch()
        output = tmp_path / "out"

        with (
            patch("core.cli_commands.execute_extraction") as mock_exec,
            patch("core.cli_commands._run_pipeline") as mock_run,
        ):
            run_extract(str(source), str(output), "every_nth_frame", 10, 1080, 0.5, True, False, False, False)

            mock_exec.assert_called_once()
            mock_run.assert_called_once()
            assert output.exists()

    def test_run_extract_fingerprint_match(self, mock_runtime, tmp_path):
        source = tmp_path / "test.mp4"
        source.touch()
        output = tmp_path / "out"
        output.mkdir()

        with (
            patch("core.cli_commands.load_fingerprint", return_value={"created_at": "now"}),
            patch("core.cli_commands.create_fingerprint", return_value={"created_at": "now"}),
            patch("core.cli_commands.fingerprints_match", return_value=True),
        ):
            # Should return early without calling execute_extraction
            with patch("core.cli_commands.execute_extraction") as mock_exec:
                run_extract(str(source), str(output), "nth", 10, 1080, 0.5, True, False, False, False)
                mock_exec.assert_not_called()

    def test_run_analyze(self, mock_runtime, tmp_path):
        source = tmp_path / "test.mp4"
        source.touch()
        output = tmp_path / "out"
        output.mkdir()

        with (
            patch("core.cli_commands.execute_analysis_orchestrator") as mock_exec,
            patch("core.cli_commands._run_pipeline") as mock_run,
        ):
            run_analyze(str(output), str(source), None, "Automatic", False, False, False)

            mock_exec.assert_called_once()
            mock_run.assert_called_once()

    def test_run_status(self, tmp_path):
        output = tmp_path / "out"
        output.mkdir()
        (output / "frame_map.json").write_text("{}")
        (output / "metadata.db").touch()

        with patch("core.cli_commands.load_fingerprint", return_value=None):
            # Just verify it doesn't crash and prints info
            run_status(str(output))

    def test_run_filter_no_db(self, tmp_path):
        output = tmp_path / "out"
        output.mkdir()

        with pytest.raises(SystemExit) as excinfo:
            run_filter(str(output), 0.5, 0.5, True, "pHash", 5, False)
        assert excinfo.value.code == 1

    def test_run_filter_success(self, mock_runtime, tmp_path):
        output = tmp_path / "out"
        output.mkdir()
        db_path = output / "metadata.db"
        db_path.touch()

        with (
            patch("core.cli_commands.Database") as mock_db_cls,
            patch("core.cli_commands.apply_all_filters_vectorized") as mock_apply,
        ):
            mock_db = mock_db_cls.return_value
            mock_db.load_all_metadata.return_value = [{"filename": "f1.png"}]

            mock_apply.return_value = ([], [], {}, {})

            run_filter(str(output), 0.5, 0.5, True, "pHash", 5, False)

            mock_apply.assert_called_once()
            mock_db.close.assert_called_once()

    def test_run_extract_clean(self, mock_runtime, tmp_path):
        source = tmp_path / "test.mp4"
        source.touch()
        output = tmp_path / "out"
        output.mkdir()
        (output / "some_file.txt").touch()

        with (
            patch("core.cli_commands.execute_extraction"),
            patch("core.cli_commands._run_pipeline"),
            patch("core.cli_commands.shutil.rmtree") as mock_rm,
        ):
            run_extract(str(source), str(output), "nth", 10, 1080, 0.5, True, False, True, False)
            mock_rm.assert_called_once_with(output)

    def test_run_filter_empty_db(self, mock_runtime, tmp_path):
        output = tmp_path / "out"
        output.mkdir()
        db_path = output / "metadata.db"
        db_path.touch()

        with (
            patch("core.cli_commands.Database") as mock_db_cls,
        ):
            mock_db = mock_db_cls.return_value
            mock_db.load_all_metadata.return_value = []

            # Should return gracefully
            run_filter(str(output), 0.5, 0.5, True, "pHash", 5, False)

    def test_run_status_missing_session(self, tmp_path):
        output = tmp_path / "nonexistent"
        # Should not crash
        run_status(str(output))

    def test_run_full(self, mock_runtime, tmp_path):
        source = tmp_path / "test.mp4"
        source.touch()
        output = tmp_path / "out"

        with (
            patch("core.cli_commands.execute_full_pipeline") as mock_exec,
            patch("core.cli_commands._run_pipeline") as mock_run,
        ):
            run_full(str(source), str(output), None, 10, 1080, False, False, False, False)

            mock_exec.assert_called_once()
            mock_run.assert_called_once()
