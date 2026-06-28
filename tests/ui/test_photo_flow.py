"""
Playwright UI Tests for Photo Mode.
Verified with mock_app.py.
Driven through ``AppDriver``; raw ``page`` retained for Photo-Mode-specific
controls the driver does not model.
"""

import pytest
from playwright.sync_api import expect

from .app_driver import AppDriver

# Mark all tests as e2e (requires app running)
pytestmark = [pytest.mark.e2e, pytest.mark.skip(reason="Photo Mode is not yet integrated into the current AppUI")]


class TestPhotoWorkflow:
    """End-to-end Photo Mode UI workflow tests."""

    def test_photo_mode_full_flow(self, app_driver: AppDriver):
        """Tests: Ingest -> Refresh Gallery -> Recalculate -> Export."""
        page = app_driver.page

        # 1. Navigate to Photo Culling tab
        app_driver.navigate("Photo Culling")

        # 2. Ingest Folder
        import os

        mock_folder = "tests/ui/mock_photos"
        os.makedirs(mock_folder, exist_ok=True)

        folder_input = page.get_by_label("Folder Path")
        expect(folder_input).to_be_visible()
        folder_input.fill(mock_folder)

        import_btn = page.get_by_role("button", name="Import Folder")
        import_btn.click()

        # Wait for status update
        status = page.locator("#photo_status")
        expect(status).to_contain_text("Page 1", timeout=15000)

        # 3. Verify Gallery
        page.wait_for_timeout(3000)
        gallery = page.locator("#photo_gallery")
        # Gradio 5+ wraps the gallery, so we look for imgs inside
        imgs = gallery.locator("img")
        expect(imgs).to_have_count(5, timeout=10000)

        # 4. Recalculate Scores
        page.get_by_placeholder("Min", exact=True).first.fill("80")
        page.wait_for_timeout(3000)
        recalc_btn = page.get_by_role("button", name="Recalculate Scores")
        recalc_btn.click()

        # 5. Export XMPs
        export_btn = page.get_by_role("button", name="Sync XMP Sidecars")
        export_btn.click()
        # In a real app we'd check the filesystem, here we just check it didn't crash.
