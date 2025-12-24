
import pytest
import numpy as np
import torch
import sys
from unittest.mock import patch, MagicMock

# We need to ensure we can import the module under test
# even if dependencies are missing, because we want to test the fallbacks.
# But pytest startup might have already imported some things.

# We will use patch.dict to mock sys.modules for the tests.

def test_edt_triton_fallback():
    from core.sam3_patches import edt_triton_fallback

    # Create a simple 3D tensor (B, H, W)
    B, H, W = 2, 10, 10
    data = torch.zeros((B, H, W), dtype=torch.uint8)

    # Set some pixels to 1 (background/target for distance transform)
    # cv2.distanceTransform calculates distance to nearest zero pixel.

    # Let's set a center pixel to 0 and others to 1
    data[:] = 1
    data[:, 5, 5] = 0

    # The function expects a tensor
    output = edt_triton_fallback(data)

    assert output.dim() == 3
    assert output.shape == (B, H, W)
    assert output.dtype == torch.float32

    # Check center is 0
    assert torch.all(output[:, 5, 5] == 0)

    # Check neighbors are 1 (distance 1)
    assert torch.all(output[:, 5, 6] == 1)
    assert torch.all(output[:, 4, 5] == 1)

def test_connected_components_fallback():
    # We need to ensure skimage is available for this test
    try:
        import skimage.measure
    except ImportError:
        pytest.skip("skimage not available")

    from core.sam3_patches import connected_components_fallback

    # Create a 4D tensor (B, 1, H, W)
    B, H, W = 2, 10, 10
    input_tensor = torch.zeros((B, 1, H, W), dtype=torch.uint8)

    # Batch 0: Two components
    # Component 1: 2 pixels
    input_tensor[0, 0, 1, 1] = 1
    input_tensor[0, 0, 1, 2] = 1

    # Component 2: 1 pixel
    input_tensor[0, 0, 5, 5] = 1

    # Batch 1: One component
    input_tensor[1, 0, 3, 3] = 1

    labels, counts = connected_components_fallback(input_tensor)

    assert labels.shape == (B, 1, H, W)
    assert counts.shape == (B, 1, H, W)

    # Check Batch 0
    l1 = labels[0, 0, 1, 1]
    l2 = labels[0, 0, 5, 5]
    assert l1 != 0
    assert l2 != 0
    assert l1 != l2

    # Counts should match component sizes
    assert counts[0, 0, 1, 1] == 2
    assert counts[0, 0, 1, 2] == 2
    assert counts[0, 0, 5, 5] == 1

    # Check Batch 1
    assert labels[1, 0, 3, 3] != 0
    assert counts[1, 0, 3, 3] == 1

def test_connected_components_fallback_3d_input():
    try:
        import skimage.measure
    except ImportError:
        pytest.skip("skimage not available")

    from core.sam3_patches import connected_components_fallback

    # Test handling of 3D input (B, H, W) implicitly unsqueezed
    B, H, W = 1, 5, 5
    input_tensor = torch.zeros((B, H, W), dtype=torch.uint8)
    input_tensor[0, 2, 2] = 1

    labels, counts = connected_components_fallback(input_tensor)

    assert labels.shape == (B, 1, H, W)
    assert counts.shape == (B, 1, H, W)
    assert counts[0, 0, 2, 2] == 1

def test_apply_patches_triton_missing():
    # We need to simulate ImportError when importing triton
    # And we need to ensure apply_patches can import sam3 modules to patch them

    from core.sam3_patches import apply_patches, edt_triton_fallback, connected_components_fallback

    # Create mock sam3 modules
    mock_edt = MagicMock()
    mock_cc = MagicMock()

    # We use patch.dict on sys.modules to:
    # 1. make 'triton' import fail (set to None usually triggers ImportError in some loaders,
    #    but here we might need to be more clever or use side_effect on import if possible.
    #    However, patch.dict won't easily simulate ImportError for an import statement
    #    unless we remove it and make sure the loader fails.
    #    Actually, if we map 'triton' to None, `import triton` might not raise ImportError,
    #    it might import None.

    # Better approach: patch builtins.__import__? Too risky.
    # The core/sam3_patches.py does:
    # try:
    #     import triton
    # except ImportError:

    # If we ensure 'triton' is NOT in sys.modules, and we can hook into import mechanism...
    # But simpler: we can patch sys.modules so that 'triton' is missing,
    # AND verify that `import triton` raises ImportError.

    # But if triton is actually installed in the environment, it will be found.
    # We can try to use `unittest.mock.patch.dict(sys.modules)` but that only affects `sys.modules` lookups.
    # If it's not in `sys.modules`, python looks in paths.

    # So we need to force `import triton` to fail.

    # Let's use a side_effect on __import__ but scoped.
    # But apply_patches is a simple function.

    # If we modify apply_patches in the file to accept an injection or something? No, I should test as is.

    # Let's try to mock `importlib.import_module`? No, `import` statement uses internal mechanism.

    # We can assume that if we mock `sys.modules['triton'] = None`, the `import triton` *might* not raise ImportError,
    # but `from core.sam3_patches import apply_patches` has already defined the function.

    # Wait, `apply_patches` executes `import triton` at runtime (inside the function).

    with patch.dict(sys.modules):
        # Remove triton if it exists
        if 'triton' in sys.modules:
            del sys.modules['triton']

        # We need to make sure the import fails.
        # One way is to set sys.modules['triton'] to a special object that raises ImportError on access?
        # No.

        # A common trick is to use `sys.modules['triton'] = None` but behavior varies.

        # Alternative: We can patch `builtins.__import__` just for this call.

        real_import = __import__
        def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == 'triton':
                raise ImportError("No triton")
            return real_import(name, globals, locals, fromlist, level)

        # We need to construct a package hierarchy for sam3 so that imports work.
        mock_sam3 = MagicMock()
        mock_sam3_model = MagicMock()
        mock_sam3_perflib = MagicMock()

        # Link them up
        mock_sam3.model = mock_sam3_model
        mock_sam3.perflib = mock_sam3_perflib
        mock_sam3_model.edt = mock_edt
        mock_sam3_perflib.connected_components = mock_cc

        with patch('builtins.__import__', side_effect=mock_import):
            with patch.dict(sys.modules, {
                'sam3': mock_sam3,
                'sam3.model': mock_sam3_model,
                'sam3.perflib': mock_sam3_perflib,
                'sam3.model.edt': mock_edt,
                'sam3.perflib.connected_components': mock_cc
            }):
                apply_patches()

                # Check if the attribute was set to the function
                assert mock_edt.edt_triton == edt_triton_fallback
                assert mock_cc.connected_components == connected_components_fallback

def test_apply_patches_triton_present():
    from core.sam3_patches import apply_patches, edt_triton_fallback, connected_components_fallback

    # Simulate triton being present
    mock_triton = MagicMock()

    with patch.dict(sys.modules, {'triton': mock_triton}):
        mock_edt = MagicMock()
        mock_cc = MagicMock()

        with patch.dict(sys.modules, {
            'sam3.model.edt': mock_edt,
            'sam3.perflib.connected_components': mock_cc
        }):
            apply_patches()

            # Should NOT have been patched
            if hasattr(mock_edt, 'edt_triton'):
                assert mock_edt.edt_triton != edt_triton_fallback
            if hasattr(mock_cc, 'connected_components'):
                assert mock_cc.connected_components != connected_components_fallback
