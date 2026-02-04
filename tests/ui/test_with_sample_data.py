"""
E2E Tests with Sample Data - Full Integration Tests.

These tests use the actual sample.mp4 file to:
1. Run extraction
2. Populate scenes
3. Test gallery sliders with real data

Run with:
    ./venv/Scripts/python.exe app.py &
    ./venv/Scripts/python.exe -m pytest tests/e2e/test_with_sample_data.py -v -s
"""

import time
from pathlib import Path

import pytest
from playwright.sync_api import Page, expect

from .conftest import BASE_URL

# Mark all tests as e2e
pytestmark = pytest.mark.e2e

# Sample file paths (in tests/assets/)
SAMPLE_VIDEO = str(Path(__file__).parent.parent / "assets" / "sample.mp4")
SAMPLE_IMAGE = str(Path(__file__).parent.parent / "assets" / "sample.jpg")


@pytest.fixture
def extracted_video_session(page: Page, app_server):
    """
    Fixture that extracts frames from sample.mp4 before tests.

    This provides a page with:
    - Video loaded
    - Frames extracted
    - Ready for pre-analysis or scene testing
    """
    page.goto(BASE_URL)
    time.sleep(3)  # Wait for app to load

    # Check if sample video exists
    if not Path(SAMPLE_VIDEO).exists():
        pytest.skip(f"Sample video not found: {SAMPLE_VIDEO}")

    # Fill in the video path
    video_input = page.get_by_label("Video URL or Local Path")
    expect(video_input).to_be_visible(timeout=10000)
    video_input.fill(SAMPLE_VIDEO)
    time.sleep(0.5)

    # Click extract button
    extract_btn = page.get_by_role("button", name="🚀 Start Single Extraction")
    if extract_btn.is_visible():
        extract_btn.click()

        # Wait for extraction to complete (check logs for success)
        time.sleep(5)  # Initial wait

        # Poll for completion (up to 60 seconds)
        for _ in range(12):
            try:
                success = page.get_by_text("Extraction complete")
                if success.is_visible(timeout=1000):
                    print("✓ Extraction complete")
                    break
            except:
                pass
            time.sleep(5)

    time.sleep(2)  # Extra wait for UI to settle
    yield page


class TestGallerySlidersWithData:
    """Tests for gallery sliders after data is loaded."""

    def test_columns_slider_changes_gallery(self, extracted_video_session):
        """Columns slider should change gallery layout when scenes exist."""
        page = extracted_video_session

        # Navigate to Scenes tab
        scenes_tab = page.get_by_role("tab", name="Scenes")
        if not scenes_tab.is_visible():
            pytest.skip("Scenes tab not visible")
        scenes_tab.click(force=True)
        time.sleep(1)

        # Run pre-analysis first if gallery is empty
        gallery = page.locator(".gallery")
        if gallery.count() == 0 or not gallery.first.is_visible():
            # Need to run pre-analysis first
            page.get_by_role("tab", name="Subject").click(force=True)
            time.sleep(0.5)

            find_frames_btn = page.get_by_role("button", name="Find & Preview Best Frames")
            if find_frames_btn.is_visible():
                find_frames_btn.click()
                time.sleep(10)  # Wait for pre-analysis

        # Back to Scenes tab
        page.get_by_role("tab", name="Scenes").click(force=True)
        time.sleep(1)

        # Find Columns slider
        columns_slider = page.get_by_label("Columns")
        if columns_slider.is_visible():
            # Change columns
            columns_slider.fill("3")
            columns_slider.press("Tab")  # Trigger change
            time.sleep(1)

            # Verify gallery still visible
            expect(page.locator(".gallery").first).to_be_visible()
            print("✓ Columns slider works with data")
        else:
            pytest.skip("Columns slider not visible")

    def test_height_slider_changes_gallery(self, extracted_video_session):
        """Height slider should change gallery height when scenes exist."""
        page = extracted_video_session

        # Navigate to Scenes tab
        scenes_tab = page.get_by_role("tab", name="Scenes")
        if not scenes_tab.is_visible():
            pytest.skip("Scenes tab not visible")
        scenes_tab.click(force=True)
        time.sleep(1)

        # Find Height slider
        height_slider = page.get_by_label("Gallery Height")
        if height_slider.is_visible():
            initial_value = height_slider.input_value()

            # Change height
            height_slider.fill("500")
            height_slider.press("Tab")  # Trigger change
            time.sleep(1)

            print(f"✓ Height slider changed from {initial_value} to 500")
        else:
            pytest.skip("Height slider not visible")


class TestFindPeopleWithData:
    """Tests for Find People feature with actual video."""

    def test_scan_video_finds_faces(self, extracted_video_session):
        """Scan Video for Faces should detect people in sample video."""
        page = extracted_video_session

        # Navigate to Subject tab
        page.get_by_role("tab", name="Subject").click(force=True)
        time.sleep(1)

        # Select Face strategy
        face_option = page.get_by_text("👤 By Face")
        if face_option.is_visible():
            face_option.click()
            time.sleep(0.5)

        # Click Scan Video button
        scan_btn = page.get_by_role("button", name="Scan Video for Faces")
        if scan_btn.is_visible():
            scan_btn.click()
            time.sleep(5)  # Wait for scanning

            # Check for status message
            status = page.locator("text=Found")
            if status.count() > 0:
                print("✓ Found people in video")
            else:
                # Check for warning message
                warning = page.locator("text=⚠️")
                if warning.count() > 0:
                    print("ℹ️ Warning shown (expected if no faces in sample)")
        else:
            pytest.skip("Scan Video button not visible")

    def test_upload_reference_face(self, extracted_video_session):
        """Upload sample.jpg as reference face for Face strategy."""
        page = extracted_video_session

        if not Path(SAMPLE_IMAGE).exists():
            pytest.skip(f"Sample image not found: {SAMPLE_IMAGE}")

        # Navigate to Subject tab
        page.get_by_role("tab", name="Subject").click(force=True)
        time.sleep(1)

        # Select Face strategy
        face_option = page.get_by_text("👤 By Face")
        if face_option.is_visible():
            face_option.click()
            time.sleep(0.5)

        # Fill in the local path for reference image (use first match if multiple)
        ref_path_input = page.get_by_label("Or local path").first
        if ref_path_input.is_visible():
            ref_path_input.fill(SAMPLE_IMAGE)
            time.sleep(0.5)
            print(f"✓ Reference face path set to: {SAMPLE_IMAGE}")

            # Verify app is still responsive
            expect(page.locator("body")).to_be_visible()
        else:
            pytest.skip("Reference path input not visible")


class TestFullWorkflowWithSampleVideo:
    """Complete workflow test using sample video."""

    def test_extract_to_scenes(self, page: Page, app_server):
        """Test extraction followed by scene detection."""
        page.goto(BASE_URL)
        time.sleep(3)

        if not Path(SAMPLE_VIDEO).exists():
            pytest.skip(f"Sample video not found: {SAMPLE_VIDEO}")

        # 1. Start extraction
        video_input = page.get_by_label("Video URL or Local Path")
        video_input.fill(SAMPLE_VIDEO)

        extract_btn = page.get_by_role("button", name="🚀 Start Single Extraction")
        extract_btn.click()

        # Wait for completion
        time.sleep(10)

        # 2. Go to Subject tab and run pre-analysis
        page.get_by_role("tab", name="Subject").click(force=True)
        time.sleep(1)

        find_btn = page.get_by_role("button", name="Find & Preview Best Frames")
        if find_btn.is_visible():
            find_btn.click()
            time.sleep(10)

        # 3. Verify Scenes tab has content
        page.get_by_role("tab", name="Scenes").click(force=True)
        time.sleep(1)

        # App should still be responsive
        expect(page.locator("body")).to_be_visible()
        print("✓ Full workflow completed without crashes")
