
from playwright.sync_api import sync_playwright

def run(playwright):
    browser = playwright.chromium.launch()
    page = browser.new_page()
    page.goto("http://localhost:7860")
    page.get_by_role("tab", name="📝 4. Metrics").click()
    page.screenshot(path="jules-scratch/verification/metrics_tab.png")
    browser.close()

with sync_playwright() as playwright:
    run(playwright)
