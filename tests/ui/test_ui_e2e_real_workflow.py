import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL, Labels, Selectors, switch_to_tab, wait_for_app_ready


@pytest.mark.e2e
def test_ui_e2e_real_workflow(page: Page, live_server):
    """
    E2E test that mimics the real user workflow:
    Upload -> Extraction -> Subject Selection -> Pre-Analysis.
    """
    # 1. Boot App & Navigate
    page.goto(BASE_URL)
    wait_for_app_ready(page)

    # 2. Stage 1: Extraction
    # Navigate to Source tab (Standard is Source handle)
    switch_to_tab(page, Labels.TAB_SOURCE)

    # Use a dummy video path for the mock/test environment
    # In a real integration run, this should be a valid file.
    video_path = "dummy_video.mp4"

    source_input = page.locator(Selectors.SOURCE_INPUT)
    source_input.fill(video_path)

    # Start extraction
    extract_btn = page.locator(Selectors.START_EXTRACTION)
    expect(extract_btn).to_be_visible()
    extract_btn.click()

    # Wait for completion with a generous timeout (60s as per refined plan)
    status_locator = page.locator(Selectors.UNIFIED_STATUS)
    expect(status_locator).to_contain_text(Selectors.STATUS_SUCCESS_EXTRACTION, timeout=60000)
    print("✅ Extraction UI stage complete.")

    # 3. Stage 2: Subject Selection & Pre-Analysis
    switch_to_tab(page, Labels.TAB_SUBJECT)

    # We'll stick to 'Automatic' for simplicity in this E2E flow
    # but ensure the button is clickable
    pre_analyze_btn = page.locator(Selectors.START_PRE_ANALYSIS)
    expect(pre_analyze_btn).to_be_visible()
    pre_analyze_btn.click()

    # Wait for pre-analysis to finish
    expect(status_locator).to_contain_text(Selectors.STATUS_SUCCESS_PRE_ANALYSIS, timeout=60000)
    print("✅ Pre-Analysis UI stage complete.")

    # 4. Final verification of result elements
    # Scenes gallery should be visible now
    switch_to_tab(page, Labels.TAB_SCENES)
    expect(page.locator(Selectors.SCENE_GALLERY)).to_be_visible(timeout=10000)
    print("🎉 E2E UI Workflow verified successfully.")
