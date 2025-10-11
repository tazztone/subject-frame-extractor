import unittest
from unittest.mock import patch, MagicMock
import os
import sys
from pathlib import Path
import numpy as np
import time

# Mock torch before it's imported by utils
sys.modules['torch'] = MagicMock()

# Add app to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import utils

class TestUtils(unittest.TestCase):

    def test_sanitize_filename(self):
        """Test filename sanitization."""
        self.assertEqual(utils.sanitize_filename("test file.txt"), "test_file.txt")
        self.assertEqual(utils.sanitize_filename("a/b\\c:d*e?f\"g<h>i|j"), "a_b_c_d_e_f_g_h_i_j")
        self.assertEqual(utils.sanitize_filename("long_filename_" * 10), "long_filename_long_filename_long_filename_long_fil")

    def test_safe_execute_with_retry_success(self):
        """Test that the function returns the correct value on success."""
        func = MagicMock(return_value="success")
        result = utils.safe_execute_with_retry(func, max_retries=3, delay=0.1)
        self.assertEqual(result, "success")
        func.assert_called_once()

    def test_safe_execute_with_retry_failure(self):
        """Test that the function retries on failure and eventually raises an exception."""
        func = MagicMock(side_effect=ValueError("Test error"))
        with self.assertRaises(ValueError):
            utils.safe_execute_with_retry(func, max_retries=2, delay=0.1)
        self.assertEqual(func.call_count, 3)

    def test_to_json_safe(self):
        """Test serialization of various object types."""
        self.assertEqual(utils._to_json_safe(Path("/tmp/test.txt")), "/tmp/test.txt")
        self.assertEqual(utils._to_json_safe(np.int64(42)), 42)
        self.assertEqual(utils._to_json_safe(np.float32(3.14159265)), 3.1416)
        self.assertEqual(utils._to_json_safe(np.array([1, 2, 3])), [1, 2, 3])
        self.assertEqual(utils._to_json_safe({'a': 1, 'b': [2, 3]}), {'a': 1, 'b': [2, 3]})
        class DummyData:
            def __init__(self):
                self.x = 1
        # This is not a dataclass, so it should return the object itself
        dummy = DummyData()
        self.assertIsInstance(utils._to_json_safe(dummy), DummyData)

    def test_safe_resource_cleanup_with_torch(self):
        """Test that torch.cuda.empty_cache is called if CUDA is available."""
        with patch('app.utils.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            with utils.safe_resource_cleanup():
                pass
            mock_torch.cuda.empty_cache.assert_called_once()

    def test_safe_resource_cleanup_without_torch(self):
        """Test that torch.cuda.empty_cache is not called if CUDA is not available."""
        with patch('app.utils.torch') as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            with utils.safe_resource_cleanup():
                pass
            mock_torch.cuda.empty_cache.assert_not_called()