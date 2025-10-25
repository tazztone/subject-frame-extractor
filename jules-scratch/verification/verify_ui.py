
import time
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

def run_verification(playwright):
    # Create a dummy file to upload
    dummy_video = Path("dummy_video.mp4")
    dummy_video.touch()

    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("http://127.0.0.1:7860")

    # Wait for the app to load
    page.wait_for_load_state('domcontentloaded')
    time.sleep(5) # Additional wait for Gradio UI to settle

    # 1. Check for the four tabs
    expect(page.locator("button[data-tab-id='0']")).to_have_text("📹 1. Frame Extraction")
    expect(page.locator("button[data-tab-id='1']")).to_have_text("👩🏼‍🦰 2. Define Subject")
    expect(page.locator("button[data-tab-id='2']")).to_have_text("🎞️ 3. Scene Selection")
    expect(page.locator("button[data-tab-id='3']")).to_have_text("📊 4. Filtering & Export")

    # 2. Upload video and run frame extraction
    page.get_by_label("Video or Image Folder").set_input_files(str(dummy_video))
    page.get_by_role("button", name="Extract Frames").click()
    # Wait for extraction to complete (check for the success message or gallery population)
    expect(page.get_by_text("Frames extracted successfully.")).to_be_visible(timeout=60000)

    # 3. Define subject and find seeds (this will auto-switch to tab 2)
    page.get_by_role("button", name="Find & Preview Scene Seeds").click()
    # Wait for pre-analysis to finish and switch to the scene selection tab
    expect(page.locator("button[data-tab-id='2'].selected")).to_be_visible(timeout=120000)

    # 4. Verify Scene Selection Tab
    # Check that the gallery has images
    gallery = page.locator("#scene_gallery_with_nav")
    # Wait for at least one image to be present in the gallery.
    expect(gallery.locator("img")).to_have_count(greater_than=0, timeout=60000)

    # Click the first image in the gallery to select a scene
    gallery.locator("img").first.click()

    # 5. Verify Scene Editor UI and Bbox
    # Check that the bounding box is visible on the main scene image.
    # The bbox is drawn on a canvas element layered over the image.
    expect(page.locator("#main_scene_editor_image canvas")).to_be_visible(timeout=10000)

    # Check that the old radio buttons are gone
    expect(page.get_by_label("Seeding Method")).not_to_be_visible()

    # Check for the new accordion
    accordion = page.locator("text=Advanced Seeding (optional)")
    expect(accordion).to_be_visible()

    # Check that it's closed by default. The content should not be visible.
    expect(page.get_by_label("DINO Text Prompt")).not_to_be_visible()

    # Open the accordion and verify content
    accordion.click()
    expect(page.get_by_label("DINO Text Prompt")).to_be_visible()

    # 6. Take Screenshot
    page.screenshot(path="jules-scratch/verification/verification.png")

    print("Verification script completed successfully.")
    browser.close()

with sync_playwright() as p:
    run_verification(p)
