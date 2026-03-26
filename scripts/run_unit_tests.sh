#!/bin/bash
uv run pytest tests/unit/test_app_ui_logic.py tests/unit/test_scene_utils.py tests/unit/test_seed_selector_extended.py --cov=core --cov-report=term-missing
