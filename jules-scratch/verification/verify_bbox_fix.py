
from playwright.sync_api import sync_playwright
import time

def run_verification():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            # 1. Navigate to the app
            page.goto("http://localhost:7860")
            page.wait_for_load_state('domcontentloaded')

            # 2. Go to the "Define Subject" tab
            page.locator("button[data-tab-id='1']").click()

            # 3. Find and click the button to start seeding
            page.get_by_role("button", name="🌱 Find & Preview Scene Seeds").click()

            # 4. Wait for the scene gallery to appear and switch to the "Scene Selection" tab
            page.wait_for_selector("div.gradio-gallery", timeout=120000)
            page.locator("button[data-tab-id='2']").click()

            # 5. Click the first scene in the gallery
            page.locator(".gradio-gallery > .grid-container > .grid-item").first.click()

            # 6. Wait for the editor to update with the selected scene's information
            page.wait_for_selector("text=Editing Scene 0", timeout=20000)

            # 7. Take the verification screenshot
            page.screenshot(path="jules-scratch/verification/verification.png")

        except Exception as e:
            print(f"An error occurred: {e}")
            page.screenshot(path="jules-scratch/verification/error.png")
        finally:
            browser.close()

if __name__ == "__main__":
    run_verification()
