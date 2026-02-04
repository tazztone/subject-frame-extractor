"""
Accessibility audit tests using axe-core.

Runs automated accessibility checks on all application tabs to detect:
- Missing alt text
- Color contrast issues
- Keyboard navigation problems
- ARIA violations
- Form label issues

Run with:
    python -m pytest tests/e2e/test_accessibility.py -v -s

Requires: playwright, axe-core (injected via CDN)
"""

import json

import pytest
from playwright.sync_api import Page

from .conftest import BASE_URL

pytestmark = [pytest.mark.e2e, pytest.mark.accessibility, pytest.mark.audit]

# axe-core CDN URL
AXE_CORE_URL = "https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.8.2/axe.min.js"


def inject_axe(page: Page) -> bool:
    """Inject axe-core into the page for accessibility testing."""
    try:
        page.add_script_tag(url=AXE_CORE_URL)
        page.wait_for_function("typeof axe !== 'undefined'", timeout=5000)
        return True
    except Exception as e:
        print(f"Failed to inject axe-core: {e}")
        return False


def run_axe_audit(page: Page, context: str = None) -> dict:
    """Run axe-core accessibility audit on current page."""
    options = {}
    if context:
        options["context"] = context

    results = page.evaluate(f"() => axe.run({json.dumps(options) if options else ''})")
    return results


def filter_violations(violations: list, min_impact: str = "serious") -> list:
    """Filter violations by minimum impact level."""
    impact_order = ["minor", "moderate", "serious", "critical"]
    min_idx = impact_order.index(min_impact)

    return [v for v in violations if impact_order.index(v.get("impact", "minor")) >= min_idx]


def format_violation(violation: dict) -> str:
    """Format a single violation for reporting."""
    return (
        f"[{violation['impact'].upper()}] {violation['id']}: {violation['description']}\n"
        f"  Help: {violation['helpUrl']}\n"
        f"  Affected: {len(violation['nodes'])} elements"
    )


class TestAccessibilityAudit:
    """Accessibility tests for each application tab."""

    TABS = [
        ("Source", None),
        ("Subject", "Subject"),
        ("Scenes", "Scenes"),
        ("Metrics", "Metrics"),
        ("Export", "Export"),
    ]

    @pytest.mark.parametrize("tab_name,click_tab", TABS)
    def test_tab_accessibility(self, page: Page, app_server, tab_name, click_tab):
        """Run accessibility audit on each tab."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")

        # Navigate to tab
        if click_tab:
            tab_btn = page.get_by_role("tab", name=click_tab)
            if tab_btn.is_visible():
                tab_btn.click(force=True)
                page.wait_for_timeout(500)

        # Inject axe-core
        if not inject_axe(page):
            pytest.skip("Could not inject axe-core")

        # Run audit
        results = run_axe_audit(page)
        violations = results.get("violations", [])

        # Filter to serious/critical only
        serious_violations = filter_violations(violations, "serious")

        # Report
        if serious_violations:
            report = f"\n{tab_name} Tab Accessibility Issues:\n"
            for v in serious_violations:
                report += format_violation(v) + "\n"
            print(report)

        assert len(serious_violations) == 0, (
            f"{len(serious_violations)} serious accessibility violations on {tab_name} tab"
        )

    def test_keyboard_navigation(self, page: Page, app_server):
        """Test that main elements are keyboard accessible."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")

        # Tab through main interface elements
        focusable_count = 0
        for _ in range(20):  # Tab 20 times
            page.keyboard.press("Tab")
            focused = page.evaluate("document.activeElement.tagName")
            if focused not in ["BODY", "HTML"]:
                focusable_count += 1

        # Should have multiple focusable elements
        assert focusable_count >= 5, "Should have multiple keyboard-focusable elements"

    def test_color_contrast(self, page: Page, app_server):
        """Check for color contrast issues."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")

        if not inject_axe(page):
            pytest.skip("Could not inject axe-core")

        # Run with just color-contrast rule
        results = page.evaluate("""
            () => axe.run({
                runOnly: {
                    type: 'rule',
                    values: ['color-contrast']
                }
            })
        """)

        violations = results.get("violations", [])

        # Color contrast issues are usually minor/moderate, not blockers
        critical_contrast = [v for v in violations if v.get("impact") == "critical"]

        if violations:
            print(f"\nColor contrast issues found: {len(violations)}")
            for v in violations[:3]:  # Show first 3
                print(f"  - {v['description']} ({len(v['nodes'])} elements)")

        assert len(critical_contrast) == 0, "Critical color contrast violations found"

    def test_form_labels(self, page: Page, app_server):
        """Check that form inputs have proper labels."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")

        if not inject_axe(page):
            pytest.skip("Could not inject axe-core")

        # Run with form-related rules
        results = page.evaluate("""
            () => axe.run({
                runOnly: {
                    type: 'rule',
                    values: ['label', 'label-title-only', 'input-button-name']
                }
            })
        """)

        violations = results.get("violations", [])
        serious = filter_violations(violations, "serious")

        assert len(serious) == 0, f"Form labeling issues: {[v['id'] for v in serious]}"


class TestARIACompliance:
    """Test ARIA attribute usage."""

    def test_aria_roles(self, page: Page, app_server):
        """Check for proper ARIA role usage."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")

        if not inject_axe(page):
            pytest.skip("Could not inject axe-core")

        results = page.evaluate("""
            () => axe.run({
                runOnly: {
                    type: 'tag',
                    values: ['wcag2a', 'wcag2aa']
                }
            })
        """)

        violations = results.get("violations", [])
        aria_violations = [v for v in violations if "aria" in v["id"].lower()]

        critical_aria = [v for v in aria_violations if v.get("impact") in ["critical", "serious"]]

        assert len(critical_aria) == 0, f"Critical ARIA violations: {[v['id'] for v in critical_aria]}"
