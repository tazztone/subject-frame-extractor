
import re
from playwright.sync_api import Page, expect

def test_new_filters_are_present(page: Page):
    """
    This test verifies that the new filters for eyes_open, yaw, and pitch are present in the UI.
    """
    # 1. Arrange: Go to the application's URL.
    page.goto("http://localhost:7860")

    # 2. Act: Click the "Filtering & Export" tab.
    filtering_tab = page.get_by_role("tab", name="Filtering & Export")
    filtering_tab.click()

    # 3. Assert: Check that the new filter accordions are present.
    expect(page.get_by_role("button", name="Eyes Open")).to_be_visible()
    expect(page.get_by_role("button", name="Yaw")).to_be_visible()
    expect(page.get_by_role("button", name="Pitch")).to_be_visible()

    # 4. Screenshot: Capture the final result for visual verification.
    page.screenshot(path="jules-scratch/verification/verification.png")
