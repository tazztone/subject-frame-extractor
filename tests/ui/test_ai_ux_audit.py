"""
AI-powered UX audit tests.
Standardized to use the new unified Selectors and Labels contract.
"""

import os

import pytest
from playwright.sync_api import Page

from .conftest import BASE_URL, switch_to_tab, wait_for_app_ready

try:
    from .ai_ux_analyzer import (
        Severity,
        analyze_screenshot_manual,
        analyze_screenshot_with_ai,
        generate_issue_report,
    )
    from .visual_test_utils import capture_state_screenshot

    HAS_ANALYZER = True
except ImportError:
    HAS_ANALYZER = False

from .ui_locators import Labels

pytestmark = [pytest.mark.e2e, pytest.mark.ux_audit, pytest.mark.audit, pytest.mark.slow]


@pytest.fixture
def use_ai():
    """Check if AI analysis should be used (API key available)."""
    return bool(os.environ.get("OPENAI_API_KEY"))


class TestUXAudit:
    """Run UX analysis on application states."""

    @pytest.mark.skipif(not HAS_ANALYZER, reason="ai_ux_analyzer not available")
    def test_source_tab_ux(self, page: Page, app_server, use_ai, tmp_path):
        """Audit Source tab UX."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        screenshot = capture_state_screenshot(page, "audit_source")

        if use_ai:
            issues = analyze_screenshot_with_ai(screenshot)
        else:
            issues = analyze_screenshot_manual(screenshot)

        # Save report
        report = generate_issue_report(issues, "Source Tab UX Audit")
        (tmp_path / "source_audit.md").write_text(report)

        # Check for critical issues
        critical = [i for i in issues if i.severity == Severity.CRITICAL]
        assert len(critical) == 0, f"Critical UX issues on Source tab: {[i.description for i in critical]}"

    @pytest.mark.skipif(not HAS_ANALYZER, reason="ai_ux_analyzer not available")
    def test_scenes_tab_ux(self, page: Page, app_server, use_ai, tmp_path):
        """Audit Scenes tab UX."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        switch_to_tab(page, Labels.TAB_SCENES)

        screenshot = capture_state_screenshot(page, "audit_scenes")

        if use_ai:
            issues = analyze_screenshot_with_ai(screenshot)
        else:
            issues = analyze_screenshot_manual(screenshot)

        report = generate_issue_report(issues, "Scenes Tab UX Audit")
        (tmp_path / "scenes_audit.md").write_text(report)

        critical = [i for i in issues if i.severity == Severity.CRITICAL]
        assert len(critical) == 0, f"Critical UX issues on Scenes tab: {[i.description for i in critical]}"

    @pytest.mark.skipif(not HAS_ANALYZER, reason="ai_ux_analyzer not available")
    def test_export_tab_ux(self, page: Page, app_server, use_ai, tmp_path):
        """Audit Export tab UX."""
        page.goto(BASE_URL)
        wait_for_app_ready(page)

        switch_to_tab(page, Labels.TAB_EXPORT)

        screenshot = capture_state_screenshot(page, "audit_export")

        if use_ai:
            issues = analyze_screenshot_with_ai(screenshot)
        else:
            issues = analyze_screenshot_manual(screenshot)

        report = generate_issue_report(issues, "Export Tab UX Audit")
        (tmp_path / "export_audit.md").write_text(report)

        critical = [i for i in issues if i.severity == Severity.CRITICAL]
        assert len(critical) == 0, f"Critical UX issues on Export tab: {[i.description for i in critical]}"


class TestFullAppAudit:
    """Run comprehensive audit across all tabs."""

    @pytest.mark.skipif(not HAS_ANALYZER, reason="ai_ux_analyzer not available")
    def test_full_app_ux_audit(self, page: Page, app_server, use_ai, tmp_path):
        """Comprehensive UX audit of entire application."""
        all_issues = []

        tabs = [Labels.TAB_SOURCE, Labels.TAB_SUBJECT, Labels.TAB_SCENES, Labels.TAB_METRICS, Labels.TAB_EXPORT]

        for tab in tabs:
            page.goto(BASE_URL)
            wait_for_app_ready(page)

            if tab != Labels.TAB_SOURCE:
                switch_to_tab(page, tab)

            screenshot = capture_state_screenshot(page, f"audit_{tab.lower()}")

            if use_ai:
                issues = analyze_screenshot_with_ai(screenshot)
            else:
                issues = analyze_screenshot_manual(screenshot)

            for issue in issues:
                issue.location = f"{tab} tab"
            all_issues.extend(issues)

        # Generate combined report
        report = generate_issue_report(all_issues, "Full Application UX Audit")
        report_path = tmp_path / "full_app_audit.md"
        report_path.write_text(report)
        print(f"\nFull audit report saved to: {report_path}")

        # Summary
        by_severity = {}
        for issue in all_issues:
            by_severity.setdefault(issue.severity, []).append(issue)

        print("\n📊 Audit Summary:")
        print(f"  🔴 Critical: {len(by_severity.get(Severity.CRITICAL, []))}")
        print(f"  🟠 Major: {len(by_severity.get(Severity.MAJOR, []))}")
        print(f"  🟡 Minor: {len(by_severity.get(Severity.MINOR, []))}")
        print(f"  🔵 Info: {len(by_severity.get(Severity.INFO, []))}")

        critical = by_severity.get(Severity.CRITICAL, [])
        assert len(critical) == 0, f"Found {len(critical)} critical UX issues"
