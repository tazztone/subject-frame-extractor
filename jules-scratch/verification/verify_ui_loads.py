from playwright.sync_api import sync_playwright, expect

def run_verification():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the Gradio app's default URL
        try:
            page.goto("http://127.0.0.1:7860")
        except Exception as e:
            print(f"Failed to connect to the application. Make sure it is running. Error: {e}")
            browser.close()
            return

        # Wait for the main header to be visible to ensure the page has loaded
        header = page.get_by_text("Frame Extractor & Analyzer v2.0")
        expect(header).to_be_visible(timeout=15000)

        # Take a screenshot of the initial UI
        screenshot_path = "jules-scratch/verification/verification.png"
        page.screenshot(path=screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    run_verification()