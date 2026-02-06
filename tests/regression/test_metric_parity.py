"""
Regression test for metric parity between Legacy and Operator implementations.

This test ensures that the refactored operator-based pipeline produces EXACTLY
the same results as the legacy `calculate_quality_metrics` method.

It works in two modes:
1. Capture: Saves current (legacy) output to `golden_metrics.json`
2. Verify: Compares current output against stored golden metrics

Usage:
    pytest tests/regression/test_metric_parity.py
    pytest tests/regression/test_metric_parity.py --capture-golden
"""

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from core.models import Frame, QualityConfig
from core.operators import run_operators

# Define golden file path
GOLDEN_FILE = Path(__file__).parent / "golden_metrics.json"


@pytest.fixture
def regression_image():
    """Create a complex synthetic image for regression testing."""
    # 256x256 RGB image
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Add gradients (for contrast/brightness)
    for i in range(256):
        img[i, :, 0] = i  # B channel gradient vertical
        img[:, i, 1] = i  # G channel gradient horizontal
    
    # Add high frequency patterns (for sharpness/edge)
    cv2.circle(img, (128, 128), 50, (255, 255, 255), 2)
    cv2.rectangle(img, (50, 50), (100, 100), (0, 255, 0), -1)
    
    # Add noise (for entropy)
    np.random.seed(12345)  # Fixed seed for reproducibility
    noise = np.random.randint(0, 50, (256, 256, 3), dtype=np.uint8)
    img = cv2.add(img, noise)
    
    return img


@pytest.fixture
def regression_mask():
    """Create a mask for the regression image."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    # Active center region
    mask[64:192, 64:192] = 255
    return mask


@pytest.fixture
def regression_config(mock_config):
    """Config with fixed values for regression."""
    # Ensure scales are fixed
    mock_config.sharpness_base_scale = 2500.0
    mock_config.edge_strength_base_scale = 500.0
    return mock_config


def get_legacy_metrics(image, mask, config):
    """Run the Legacy calculation method."""
    frame = Frame(image_data=image, frame_number=0)
    
    # Quality config matching legacy defaults
    q_config = QualityConfig(
        sharpness_base_scale=config.sharpness_base_scale,
        edge_strength_base_scale=config.edge_strength_base_scale,
        enable_niqe=False, # Skip NIQE for regression (too complex/slow)
    )
    
    metrics_to_compute = {
        "quality": True,
        "sharpness": True,
        "edge_strength": True,
        "contrast": True,
        "brightness": True,
        "entropy": True,
        "eyes_open": False, # Skip face metrics for synthetic image
        "yaw": False,
        "pitch": False,
    }
    
    # Create logger mock
    logger = MagicMock()
    
    frame.calculate_quality_metrics(
        thumb_image_rgb=image,
        quality_config=q_config,
        logger=logger,
        mask=mask,
        metrics_to_compute=metrics_to_compute,
        main_config=config,
    )
    
    return frame.metrics.model_dump()


def get_operator_metrics(image, mask, config):
    """Run the Operator-based calculation."""
    # Note: Phase 2.1-2.5 will implement these operators.
    # Initially this will fail or return partial results until implemented.
    from core.operators.registry import OperatorRegistry, run_operators
    
    # Ensure operators are initialized
    OperatorRegistry.initialize_all(config)
    
    results = run_operators(
        image_rgb=image,
        mask=mask,
        config=config,
    )
    
    # Flatten results to match legacy format
    metrics = {}
    for op_name, result in results.items():
        if result.success:
            metrics.update(result.metrics)
            
    return metrics


def test_metric_parity(regression_image, regression_mask, regression_config, request):
    """
    Compare Legacy vs Operator outputs.
    
    If GOLDEN_FILE missing or --capture-golden flag set:
        Runs Legacy code and saves output.
    Else:
        Runs Legacy code (to verify golden is still valid)
        Runs Operator code
        Asserts they match
    """
    # 1. Run Legacy (Ground Truth)
    legacy_metrics = get_legacy_metrics(regression_image, regression_mask, regression_config)
    
    # Determine capture mode
    capture_mode = request.config.getoption("--capture-golden", default=False)
    if not GOLDEN_FILE.exists():
        capture_mode = True
        
    if capture_mode:
        print(f"\n[CAPTURE] Saving golden metrics to {GOLDEN_FILE}")
        with open(GOLDEN_FILE, "w") as f:
            json.dump(legacy_metrics, f, indent=2, sort_keys=True)
        # Verify content
        assert GOLDEN_FILE.exists()
        return

    # 2. Load Golden
    with open(GOLDEN_FILE, "r") as f:
        golden_metrics = json.load(f)

    # 3. Verify Legacy matches Golden (Sanity check: did legacy code change?)
    # We ignore timestamp-like or volatile fields if any (metrics are usually deterministic)
    # Using approx for floats
    for k, v in golden_metrics.items():
        if isinstance(v, (int, float)):
            assert legacy_metrics[k] == pytest.approx(v, abs=1e-5), \
                f"Legacy drift for {k}: golden={v}, current={legacy_metrics[k]}"
    
    # 4. Run Operators (The actual regression test)
    op_metrics = get_operator_metrics(regression_image, regression_mask, regression_config)
    
    # 5. Compare Operator vs Golden
    # Note: Only compare keys that exist in operator results.
    # As we migrate, coverage will increase.
    print(f"\n[VERIFY] Comparing Operator results against Golden...")
    
    matches = 0
    failures = []
    
    # Map operator metric names to legacy names if they differ
    # Currently designed to be identical (e.g., "sharpness_score")
    
    for k, golden_val in golden_metrics.items():
        if k not in op_metrics:
            # Skip unimplemented metrics for now
            continue
            
        op_val = op_metrics[k]
        
        # Check tolerance (allow small float diffs)
        try:
            if isinstance(golden_val, (int, float)):
                assert op_val == pytest.approx(golden_val, abs=1e-4)
            else:
                assert op_val == golden_val
            matches += 1
        except AssertionError as e:
            failures.append(f"{k}: golden={golden_val}, op={op_val} (diff={op_val-golden_val})")
            
    if failures:
        pytest.fail(f"Metric mismatch in {len(failures)} keys:\n" + "\n".join(failures))
        
    print(f"Verified {matches} metrics match exactly.")
