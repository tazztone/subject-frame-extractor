
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from core.sam3_patches import edt_triton_fallback, connected_components_fallback

class TestSam3Patches:

    def test_edt_triton_fallback_2d_batch(self):
        # Batch size 2, 10x10 image
        B, H, W = 2, 10, 10
        data = torch.ones((B, H, W), dtype=torch.float32) # Standard input for dist transform is usually binary mask

        # Set a zero pixel in center
        data[:, 5, 5] = 0

        output = edt_triton_fallback(data)

        assert output.shape == (B, H, W)
        assert output[0, 5, 5] == 0
        assert output[0, 5, 6] == 1 # 1 pixel away

    def test_edt_triton_fallback_all_zeros(self):
        B, H, W = 1, 5, 5
        data = torch.zeros((B, H, W), dtype=torch.float32)

        output = edt_triton_fallback(data)

        # Distance to nearest zero is 0 everywhere
        assert torch.all(output == 0)

    def test_connected_components_fallback_simple(self):
        # Create (B, 1, H, W) tensor
        B, H, W = 1, 10, 10
        data = torch.zeros((B, 1, H, W), dtype=torch.uint8)

        # Two distinct components
        data[0, 0, 1, 1] = 1
        data[0, 0, 8, 8] = 1

        labels, counts = connected_components_fallback(data)

        assert labels.shape == (B, 1, H, W)
        assert counts.shape == (B, 1, H, W)

        l1 = labels[0, 0, 1, 1].item()
        l2 = labels[0, 0, 8, 8].item()

        assert l1 > 0
        assert l2 > 0
        assert l1 != l2

        assert counts[0, 0, 1, 1] == 1
        assert counts[0, 0, 8, 8] == 1

    def test_connected_components_fallback_complex(self):
        # Check larger component counts
        B, H, W = 1, 10, 10
        data = torch.zeros((B, 1, H, W), dtype=torch.uint8)

        # 3-pixel L-shape
        data[0, 0, 1, 1] = 1
        data[0, 0, 1, 2] = 1
        data[0, 0, 2, 1] = 1

        labels, counts = connected_components_fallback(data)

        lbl = labels[0, 0, 1, 1].item()
        assert labels[0, 0, 1, 2].item() == lbl
        assert labels[0, 0, 2, 1].item() == lbl

        assert counts[0, 0, 1, 1].item() == 3

    def test_connected_components_fallback_3d_input_compat(self):
        # Input (B, H, W) should be handled by unsqueezing
        B, H, W = 1, 5, 5
        data = torch.zeros((B, H, W), dtype=torch.uint8)
        data[0, 2, 2] = 1

        labels, counts = connected_components_fallback(data)

        assert labels.shape == (B, 1, H, W)
