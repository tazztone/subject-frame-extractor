import json
import shutil
import tempfile
from pathlib import Path
import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL, wait_for_app_ready, switch_to_tab, open_accordion
from .ui_locators import Selectors, Labels

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
            "tracker_model_name": "sam2",
            "best_frame_strategy": "Largest Person"
        }
        (session_path / "run_config.json").write_text(json.dumps(run_config), encoding="utf-8")

        # 2. Create scenes.json (pairs of start, end frames)
        scenes = [[0, 100], [101, 200]]
        (session_path / "scenes.json").write_text(json.dumps(scenes), encoding="utf-8")

        # 3. Create scene_seeds.json
        seeds = {
            "0": {"status": "included", "best_frame": 50, "bbox": [10, 10, 50, 50]},
            "1": {"status": "included", "best_frame": 150, "bbox": [20, 20, 60, 60]}
        }
        (session_path / "scene_seeds.json").write_text(json.dumps(seeds), encoding="utf-8")

        # 4. Create dummy metadata.db to trigger Analysis tab interactivity
        (session_path / "metadata.db").touch()

        yield str(session_path)

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_load_session_updates_ui(self, page: Page, app_server, mock_session_dir):
        """Verify that loading a session directory updates UI components correctly."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        # 1. Open Session Accordion
        open_accordion(page, Labels.SESSION_ACCORDION)

        # 2. Fill the session path input
        session_input = page.locator(Selectors.SESSION_INPUT)
        expect(session_input).to_be_visible()
        session_input.fill(mock_session_dir)

        # 3. Click Load (Force click to override any Gradio 5 overlays)
        page.get_by_role("button", name="Load Session").click(force=True)

        # 4. Verify Success Message
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text("Session Loaded.", timeout=10000)

        # 5. Verify UI Updates (Extraction Settings)
        # MUST open the advanced accordion as it's collapsed by default
        open_accordion(page, Labels.ADVANCED_ACCORDION)

        # Check Megapixels (0.8)
        mp_input = page.locator(Selectors.THUMB_MEGAPIXELS)
        expect(mp_input).to_have_value("0.8")

        # 6. Verify Analysis Tab is now interactive (since metadata.db exists)
        analysis_tab = page.get_by_role("tab", name=Labels.TAB_METRICS)
        expect(analysis_tab).to_be_enabled()

        # 7. Verify Scene Gallery is populated (2 scenes)
        switch_to_tab(page, Labels.TAB_SCENES)

        # Check gallery visibility
        gallery = page.locator(Selectors.SCENE_GALLERY)
        expect(gallery).to_be_visible()

        # Check gallery status text (MUST open Batch Filter accordion)
        open_accordion(page, Labels.BATCH_FILTER_ACCORDION)
        status_text = page.locator(Selectors.SCENE_FILTER_STATUS)
        expect(status_text).to_contain_text("2")

    def test_load_invalid_session_shows_error(self, page: Page, app_server):
        """Verify that loading a non-existent directory shows an error."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        open_accordion(page, Labels.SESSION_ACCORDION)

        session_input = page.locator(Selectors.SESSION_INPUT)
        session_input.fill("/non/existent/path")

        # 3. Click Load
        page.wait_for_timeout(500)  # Let Gradio 5 stabilize
        load_btn = page.get_by_role("button", name="Load Session")
        expect(load_btn).to_be_visible()
        load_btn.click(force=True)

        # Verify Error in Status
        expect(page.locator(Selectors.UNIFIED_STATUS)).to_contain_text("Error", timeout=10000)
        # In Gradio 5, logs might be in a different element, but Selectors.LOG_TEXTAREA should find it
        expect(page.locator(Selectors.LOG_TEXTAREA)).to_contain_text("Session directory does not exist", timeout=10000)
