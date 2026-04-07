import os
from unittest.mock import MagicMock

import pytest


@pytest.fixture(scope="module")
def module_model_registry():
    """Module-scoped real (or stub) ModelRegistry for integration tests.

    When PYTEST_INTEGRATION_MODE=true, returns a real ModelRegistry loaded
    from the environment. Otherwise returns a MagicMock that satisfies
    fixture requirements for unit-within-integration tests.
    """
    is_integration = os.environ.get("PYTEST_INTEGRATION_MODE", "false").lower() == "true"

    if is_integration:
        from core.config import Config
        from core.logger import AppLogger
        from core.managers.registry import ModelRegistry

        logger = AppLogger("integration_registry")
        config = Config()
        registry = ModelRegistry(config, logger)
        # We don't necessarily load all models here, just return the registry
        return registry
    else:
        # Fallback to mock for non-integration runs that accidentally hit integration tests
        mock = MagicMock()
        mock.get_tracker.return_value = MagicMock()
        return mock
