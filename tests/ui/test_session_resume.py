import json
import shutil
import tempfile
from pathlib import Path

import pytest
from playwright.sync_api import expect

from .app_driver import AppDriver
from .ui_locators import Labels, Selectors

# Mark as e2e test
pytestmark = pytest.mark.e2e


class TestSessionResume:
    """
    E2E tests for loading and resuming previous sessions.
    """

    @pytest.fixture
    def mock_session_dir(self):
        """Creates a temporary mock session directory."""
        temp_dir = tempfile.mkdtemp()
        session_path = Path(temp_dir)

        # 1. Create run_config.json
        run_config = {
            "source_path": "mock_video.mp4",
            "max_resolution": "720",
            "thumb_megapixels": 0.8,
            "scene_detect": True,
            "method": "scene",
            "pre_analysis_enabled": True,
            "pre_sample_nth": 2,
            "tracker_model_name": "sam3",
            "best_frame_strategy": "Largest Person",
        }
        (session_path / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")

        # 2. Create scenes.json (pairs of start, end frames)
        scenes = [[0, 100], [101, 200]]
        (session_path / "scenes.json").write_text(json.dumps(scenes), encoding="utf-8")

        # 3. Create scene_seeds.json
        seeds = {
            "0": {"status": "included", "best_frame": 50, "bbox": [10, 10, 50, 50]},
            "1": {"status": "included", "best_frame": 150, "bbox": [20, 20, 60, 60]},
        }
        (session_path / "scene_seeds.json").write_text(json.dumps(seeds), encoding="utf-8")

        # 4. Create dummy metadata.db to trigger Analysis tab interactivity
        (session_path / "metadata.db").touch()

        yield str(session_path)

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_load_session_updates_ui(self, app_driver: AppDriver, mock_session_dir):
        """Verify that loading a session directory updates UI components correctly."""
        # 1-3. Open session accordion, fill path, click Load
        app_driver.load_session(mock_session_dir)

        # 4. Verify Success Message
        app_driver.expect_status("Session Loaded.", timeout=10000)

        # 5. Verify UI Updates (Extraction Settings)
        # MUST open the advanced accordion as it's collapsed by default
        app_driver.open_accordion(Labels.ADVANCED_ACCORDION)

        # Check Megapixels (0.8)
        mp_input = app_driver.page.locator(f"{Selectors.THUMB_MEGAPIXELS} input[type='number']")
        expect(mp_input).to_have_value("0.8")

        # 6. Verify Analysis Tab is now interactive (since metadata.db exists)
        analysis_tab = app_driver.page.get_by_role("tab", name=Labels.TAB_METRICS)
        expect(analysis_tab).to_be_enabled()

        # 7. Verify Scene Gallery is populated (2 scenes)
        app_driver.navigate(Labels.TAB_SCENES)
        app_driver.expect_visible(Selectors.SCENE_GALLERY)

        # Check gallery status text (MUST open Batch Filter accordion)
        app_driver.open_accordion(Labels.BATCH_FILTER_ACCORDION)
        expect(app_driver.page.locator(Selectors.SCENE_FILTER_STATUS)).to_contain_text("2")

    def test_load_invalid_session_shows_error(self, app_driver: AppDriver):
        """Verify that loading a non-existent directory shows an error."""
        app_driver.load_session("/non/existent/path")

        # Verify Error in Status, then detail in logs.
        app_driver.expect_status("Failed", timeout=10000)
        app_driver.expect_log("does not exist", timeout=10000)
