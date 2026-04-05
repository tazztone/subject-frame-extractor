"""
Unit tests for handle_common_errors decorator.
"""

from unittest.mock import MagicMock, patch

from core.error_handling import handle_common_errors


def test_handle_common_errors_success():
    """Test handle_common_errors with successful generator."""

    @handle_common_errors
    def success_gen():
        yield {"data": 1}
        yield {"data": 2}

    results = list(success_gen())
    assert results == [{"data": 1}, {"data": 2}]


def test_handle_common_errors_generic_exception():
    """Test handle_common_errors with generic Exception."""
    mock_logger = MagicMock()

    @handle_common_errors
    def fail_gen(logger=None):
        yield {"data": 1}
        raise Exception("Generic failure")

    results = list(fail_gen(logger=mock_logger))

    # Should yield success then error
    assert len(results) == 2
    assert results[0] == {"data": 1}
    assert results[1]["success"] is False
    assert "Generic failure" in results[1]["error"]
    assert results[1]["done"] is False
    assert results[1]["status_message"] is not None

    mock_logger.error.assert_called()


def test_handle_common_errors_file_not_found():
    """Test handle_common_errors with FileNotFoundError."""

    @handle_common_errors
    def fail_gen():
        raise FileNotFoundError("config.json missing")

    results = list(fail_gen())
    assert len(results) == 1
    assert "File not found" in results[0]["status_message"]
    assert "config.json missing" in results[0]["error"]


def test_handle_common_errors_value_error():
    """Test handle_common_errors with ValueError."""

    @handle_common_errors
    def fail_gen():
        raise ValueError("Invalid setting")

    results = list(fail_gen())
    assert len(results) == 1
    assert "Invalid argument" in results[0]["status_message"]


def test_handle_common_errors_cuda_oom():
    """Test handle_common_errors with CUDA OutOfMemoryError."""
    # We need to mock torch.cuda.OutOfMemoryError because it's caught in the decorator
    from tests.helpers.exceptions import OutOfMemoryError

    with patch("torch.cuda.OutOfMemoryError", OutOfMemoryError, create=True):

        @handle_common_errors
        def fail_gen():
            raise OutOfMemoryError("Alloc failed")

        results = list(fail_gen())
        assert len(results) == 1
        assert "GPU memory error" in results[0]["status_message"]
        assert results[0]["success"] is False


def test_handle_common_errors_runtime_error():
    """Test handle_common_errors with RuntimeError."""

    @handle_common_errors
    def fail_gen():
        raise RuntimeError("Something crashed")

    results = list(fail_gen())
    assert len(results) == 1
    assert "Runtime error" in results[0]["status_message"]
