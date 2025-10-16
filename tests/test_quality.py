import unittest
import numpy as np
from app.quality import compute_entropy

class TestQuality(unittest.TestCase):
    def test_compute_entropy_single_value(self):
        # Histogram with a single value has zero entropy
        hist = np.zeros(256, dtype=np.uint64)
        hist[128] = 100
        entropy = compute_entropy(hist)
        self.assertAlmostEqual(entropy, 0.0, places=5)

    def test_compute_entropy_uniform_distribution(self):
        # A uniform distribution has maximum entropy, which is log2(256) = 8
        hist = np.ones(256, dtype=np.uint64)
        entropy = compute_entropy(hist)
        self.assertAlmostEqual(entropy, 8.0, places=5)

if __name__ == '__main__':
    unittest.main()