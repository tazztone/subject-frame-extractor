"""Image quality assessment utilities."""

import numpy as np
from numba import njit


@njit
def compute_entropy(hist):
    """Compute entropy from a histogram using Numba for performance."""
    prob = hist / (np.sum(hist) + 1e-7)
    entropy = -np.sum(prob[prob > 0] * np.log2(prob[prob > 0]))
    return min(max(entropy / 8.0, 0), 1.0)
