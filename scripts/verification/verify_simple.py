from playwright.sync_api import sync_playwright
import time

def verify_ui_simple():
    print("Starting Playwright...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # --- ADD THIS BLOCK ---
        # Print all browser console logs to your terminal
        page.on("console", lambda msg: print(f"BROWSER CONSOLE: {msg.text}"))
        page.on("pageerror", lambda exc: print(f"BROWSER ERROR: {exc}"))
        # ----------------------

        print("Navigating...")
        try:
            # Increase timeout to 60s
            page.goto("http://127.0.0.1:7860", timeout=60000, wait_until="domcontentloaded")
            print("Navigation complete.")
        except Exception as e:
            print(f"Navigation failed: {e}")
            return

        print("Waiting for 10 seconds...")
        time.sleep(10)

        print("Taking screenshot...")
        from pathlib import Path
        output_path = Path(__file__).parent / "simple_check.png"
        page.screenshot(path=str(output_path))
        print(f"Screenshot saved to {output_path}")

        title = page.title()
        print(f"Page Title: {title}")

        browser.close()

if __name__ == "__main__":
    verify_ui_simple()
