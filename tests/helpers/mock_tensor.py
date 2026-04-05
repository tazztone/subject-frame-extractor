"""
Robust mock tensor factory for unit tests.
This helper is standalone to avoid closure/recursion issues in conftest.
"""

from unittest.mock import MagicMock


def _create_mock_tensor(name="tensor", shape=None, device_mock=None, dtype_mock=None, depth=0):
    """Factory for robust mock tensors that support .cpu().numpy() and basic torch ops."""
    if depth > 2:
        # Stop recursion
        m = MagicMock(name=f"{name}_leaf")
        m.item.return_value = 1.0
        m.__bool__.return_value = True
        return m

    m = MagicMock(name=name)

    # Core methods
    m.cpu.return_value = m
    m.cuda.return_value = m
    m.detach.return_value = m
    m.to.return_value = m
    m.float.return_value = m
    m.long.return_value = m
    m.half.return_value = m
    m.int.return_value = m
    m.bool.return_value = m

    # Use a simple side_effect for type to avoid complex logic
    def mock_type(*args):
        if not args:
            return dtype_mock
        return m

    m.type.side_effect = mock_type

    # Normalize shape to tuple if integer
    if isinstance(shape, int):
        shape = (shape,)

    m.shape = shape
    m.ndim = len(shape) if shape is not None else 0
    m.device = device_mock
    m.dtype = dtype_mock

    # Calculate size
    import math

    s = math.prod(shape) if shape is not None else 10000  # Default large size
    m.size = s
    m.numel.return_value = s
    m.nelement.return_value = s

    # Boolean/Numeric protocols
    m.__bool__.return_value = True
    m.__len__.return_value = shape[0] if (shape and len(shape) > 0) else 1
    m.__float__ = MagicMock(return_value=1.0)
    m.__int__ = MagicMock(return_value=1)
    m.__index__ = MagicMock(return_value=1)

    # Numpy/Array protocol
    def mock_array(*args, **kwargs):
        import numpy as np

        return np.zeros(shape if shape else (100, 100), dtype=np.float32)

    m.__array__ = MagicMock(side_effect=mock_array)

    def mock_numpy():
        import numpy as np

        return np.zeros(shape if shape else (100, 100), dtype=np.float32)

    m.numpy.side_effect = mock_numpy

    # Slicing
    def mock_getitem(idx):
        return _create_mock_tensor(
            name="sliced_tensor",
            shape=shape[1:] if (shape and len(shape) > 0) else None,
            device_mock=device_mock,
            dtype_mock=dtype_mock,
            depth=depth + 1,
        )

    m.__getitem__.side_effect = mock_getitem

    # Operators (Use fixed return names to prevent recursive f-strings)
    m.__gt__.side_effect = lambda other: _create_mock_tensor("cmp_result", shape, device_mock, dtype_mock, depth + 1)
    m.__lt__.side_effect = lambda other: _create_mock_tensor("cmp_result", shape, device_mock, dtype_mock, depth + 1)
    m.__ge__.side_effect = lambda other: _create_mock_tensor("cmp_result", shape, device_mock, dtype_mock, depth + 1)
    m.__le__.side_effect = lambda other: _create_mock_tensor("cmp_result", shape, device_mock, dtype_mock, depth + 1)
    m.__add__.side_effect = lambda other: _create_mock_tensor("calc_result", shape, device_mock, dtype_mock, depth + 1)
    m.__sub__.side_effect = lambda other: _create_mock_tensor("calc_result", shape, device_mock, dtype_mock, depth + 1)
    m.__mul__.side_effect = lambda other: _create_mock_tensor("calc_result", shape, device_mock, dtype_mock, depth + 1)
    m.__truediv__.side_effect = lambda other: _create_mock_tensor(
        "calc_result", shape, device_mock, dtype_mock, depth + 1
    )

    # Reductions
    m.any.return_value = True
    m.all.return_value = True
    m.sum.side_effect = lambda *a, **k: _create_mock_tensor("scalar_result", (), device_mock, dtype_mock, depth + 1)
    m.mean.side_effect = lambda *a, **k: _create_mock_tensor("scalar_result", (), device_mock, dtype_mock, depth + 1)
    m.max.side_effect = lambda *a, **k: _create_mock_tensor("scalar_result", (), device_mock, dtype_mock, depth + 1)
    m.min.side_effect = lambda *a, **k: _create_mock_tensor("scalar_result", (), device_mock, dtype_mock, depth + 1)
    m.item.return_value = 1.0

    # Formatting to prevent further recursion - use fixed strings
    m.__repr__ = lambda self=m: f"<MockTensor {name}>"
    m.__str__ = lambda self=m: f"<MockTensor {name}>"

    return m


def create_mock_tensor(name="tensor", shape=None, device_mock=None, dtype_mock=None):
    """Public API for mock tensor factory."""
    return _create_mock_tensor(name=name, shape=shape, device_mock=device_mock, dtype_mock=dtype_mock, depth=0)
