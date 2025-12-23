"""
AI-powered UX audit tests.

Uses the ai_ux_analyzer module to detect UX issues in screenshots.
Can run with or without AI API - manual mode uses heuristic checks.

Run with:
    python -m pytest tests/e2e/test_ai_ux_audit.py -v -s

For AI-powered analysis:
    OPENAI_API_KEY=sk-xxx python -m pytest tests/e2e/test_ai_ux_audit.py -v -s
"""
import pytest
from playwright.sync_api import Page
import os

from .conftest import BASE_URL

try:
    from .visual_test_utils import capture_state_screenshot
    from .ai_ux_analyzer import (
        analyze_screenshot_manual,
        analyze_screenshot_with_ai,
        generate_issue_report,
        UXIssue,
        Severity
    )
    HAS_ANALYZER = True
except ImportError:
    HAS_ANALYZER = False

pytestmark = [pytest.mark.e2e, pytest.mark.ux_audit]


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
        page.wait_for_load_state("networkidle")
        
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
        """Audit Scenes tab UX - where pagination issues were found."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        
        page.get_by_role("tab", name="Scenes").click(force=True)
        page.wait_for_timeout(500)
        
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
        """Audit Export tab UX - filter controls and results display."""
        page.goto(BASE_URL)
        page.wait_for_load_state("networkidle")
        
        page.get_by_role("tab", name="Export").click(force=True)
        page.wait_for_timeout(500)
        
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
        
        tabs = ["Source", "Subject", "Scenes", "Metrics", "Export"]
        
        for tab in tabs:
            page.goto(BASE_URL)
            page.wait_for_load_state("networkidle")
            
            if tab != "Source":
                tab_btn = page.get_by_role("tab", name=tab)
                if tab_btn.is_visible():
                    tab_btn.click(force=True)
                    page.wait_for_timeout(500)
            
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
        
        print(f"\n📊 Audit Summary:")
        print(f"  🔴 Critical: {len(by_severity.get(Severity.CRITICAL, []))}")
        print(f"  🟠 Major: {len(by_severity.get(Severity.MAJOR, []))}")
        print(f"  🟡 Minor: {len(by_severity.get(Severity.MINOR, []))}")
        print(f"  🔵 Info: {len(by_severity.get(Severity.INFO, []))}")
        
        critical = by_severity.get(Severity.CRITICAL, [])
        assert len(critical) == 0, f"Found {len(critical)} critical UX issues"
