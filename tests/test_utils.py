import pytest
import sys
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from unittest.mock import patch, MagicMock

# Import the code to be tested
from app.utils import (
    sanitize_filename,
    safe_resource_cleanup,
    _to_json_safe,
)

# --- Tests for sanitize_filename ---

def test_sanitize_filename_removes_invalid_chars():
    assert sanitize_filename("a!b@c#d$e%f^g&h*i(j)k.txt") == "a_b_c_d_e_f_g_h_i_j_k.txt"

def test_sanitize_filename_truncates():
    long_name = "a" * 100 + ".txt"
    sanitized = sanitize_filename(long_name)
    assert len(sanitized) == 50
    assert sanitized.endswith('a')

def test_sanitize_filename_allows_valid_chars():
    valid_name = "valid-file_name.123.zip"
    assert sanitize_filename(valid_name) == valid_name

# --- Tests for safe_resource_cleanup ---

@patch('app.utils.gc.collect')
@patch('app.utils.torch')
def test_safe_resource_cleanup(mock_torch, mock_gc_collect):
    """Test cleanup when CUDA is available."""
    mock_torch.cuda.is_available.return_value = True

    with safe_resource_cleanup():
        pass

    mock_gc_collect.assert_called_once()
    mock_torch.cuda.empty_cache.assert_called_once()

@patch('app.utils.gc.collect')
@patch('app.utils.torch')
def test_safe_resource_cleanup_no_cuda(mock_torch, mock_gc_collect):
    """Test cleanup when CUDA is not available."""
    mock_torch.cuda.is_available.return_value = False

    with safe_resource_cleanup():
        pass

    mock_gc_collect.assert_called_once()
    mock_torch.cuda.empty_cache.assert_not_called()

# --- Tests for _to_json_safe ---

def test_to_json_safe_dict():
    assert _to_json_safe({'a': 1, 'b': 2.0}) == {'a': 1, 'b': 2.0}

def test_to_json_safe_list():
    assert _to_json_safe([1, 2.0, "three"]) == [1, 2.0, "three"]

def test_to_json_safe_path_object():
    p = Path("/tmp/test.txt")
    assert _to_json_safe(p) == "/tmp/test.txt"

def test_to_json_safe_numpy_generic():
    assert _to_json_safe(np.int64(42)) == 42
    assert isinstance(_to_json_safe(np.int64(42)), int)
    assert _to_json_safe(np.float32(3.14159265)) == 3.1416

def test_to_json_safe_numpy_array():
    arr = np.array([1, 2, 3])
    assert _to_json_safe(arr) == [1, 2, 3]

def test_to_json_safe_float_rounding():
    assert _to_json_safe(3.1415926535) == 3.1416

@dataclass
class MyDataClass:
    x: int
    y: str

def test_to_json_safe_dataclass():
    instance = MyDataClass(x=1, y="test")
    expected = {'x': 1, 'y': 'test'}
    assert _to_json_safe(instance) == expected

def test_to_json_safe_nested_structure():
    p = Path("/path/to/file")
    @dataclass
    class NestedDc:
        path: Path

    data = {
        'list': [1, np.float32(2.5), p],
        'dc': NestedDc(path=p),
        'np_array': np.array([4, 5])
    }

    expected = {
        'list': [1, 2.5, '/path/to/file'],
        'dc': {'path': '/path/to/file'},
        'np_array': [4, 5]
    }

    assert _to_json_safe(data) == expected