"""
AI-powered UX analysis using screenshot inspection.

This module provides tools for analyzing UI screenshots against a UX checklist.
It can be integrated with vision AI APIs (GPT-4V, Claude, etc.) for automated analysis.

Usage:
    from ai_ux_analyzer import analyze_screenshot, UX_CHECKLIST
    issues = analyze_screenshot(screenshot_path)
"""
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class Severity(Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


class Category(Enum):
    LAYOUT = "layout"
    USABILITY = "usability"
    ACCESSIBILITY = "accessibility"
    FEEDBACK = "feedback"
    CONTROLS = "controls"
    CONSISTENCY = "consistency"


@dataclass
class UXIssue:
    """Represents a detected UX issue."""
    category: Category
    severity: Severity
    element: str
    description: str
    suggestion: str
    location: Optional[str] = None


@dataclass 
class UXCheckItem:
    """A single item in the UX checklist."""
    category: Category
    question: str
    severity_if_violated: Severity = Severity.MAJOR


# Comprehensive UX checklist for UI analysis
UX_CHECKLIST: List[UXCheckItem] = [
    # Layout & Visual
    UXCheckItem(Category.LAYOUT, "Are all elements properly aligned?"),
    UXCheckItem(Category.LAYOUT, "Is there sufficient spacing between elements?"),
    UXCheckItem(Category.LAYOUT, "Are similar elements styled consistently?"),
    UXCheckItem(Category.LAYOUT, "Is the visual hierarchy clear?"),
    
    # Usability
    UXCheckItem(Category.USABILITY, "Are interactive elements clearly identifiable?", Severity.CRITICAL),
    UXCheckItem(Category.USABILITY, "Is the current state/selection visible?", Severity.CRITICAL),
    UXCheckItem(Category.USABILITY, "Are error states clearly communicated?", Severity.CRITICAL),
    UXCheckItem(Category.USABILITY, "Is pagination intuitive (buttons vs text fields)?"),
    UXCheckItem(Category.USABILITY, "Are labels descriptive and helpful?"),
    
    # Feedback
    UXCheckItem(Category.FEEDBACK, "Is there visible feedback for user actions?", Severity.MAJOR),
    UXCheckItem(Category.FEEDBACK, "Are loading states shown for async operations?"),
    UXCheckItem(Category.FEEDBACK, "Are logs/status messages visible and readable?"),
    UXCheckItem(Category.FEEDBACK, "Are success/error messages clear?"),
    
    # Controls
    UXCheckItem(Category.CONTROLS, "Do sliders have clear labels and current values?"),
    UXCheckItem(Category.CONTROLS, "Are disabled elements visually distinct?"),
    UXCheckItem(Category.CONTROLS, "Do dropdowns show current selection?"),
    UXCheckItem(Category.CONTROLS, "Are buttons appropriately sized for their importance?"),
    
    # Accessibility
    UXCheckItem(Category.ACCESSIBILITY, "Is text readable (sufficient contrast)?", Severity.CRITICAL),
    UXCheckItem(Category.ACCESSIBILITY, "Are interactive elements keyboard accessible?"),
    UXCheckItem(Category.ACCESSIBILITY, "Are form fields properly labeled?"),
    
    # Consistency
    UXCheckItem(Category.CONSISTENCY, "Are icons used consistently?"),
    UXCheckItem(Category.CONSISTENCY, "Is terminology consistent across the UI?"),
    UXCheckItem(Category.CONSISTENCY, "Are action button styles consistent?"),
]


def analyze_screenshot_manual(screenshot_path: Path) -> List[UXIssue]:
    """
    Analyze screenshot for UX issues using rule-based checks.
    
    This is a placeholder for manual/heuristic analysis.
    For actual AI-powered analysis, use analyze_screenshot_with_ai().
    """
    issues = []
    # Placeholder - manual inspection would happen here
    return issues


def analyze_screenshot_with_ai(
    screenshot_path: Path,
    api_key: Optional[str] = None,
    model: str = "gpt-4-vision-preview"
) -> List[UXIssue]:
    """
    Analyze screenshot using vision AI API.
    
    Args:
        screenshot_path: Path to the screenshot image
        api_key: API key for the vision service (or uses env var)
        model: Model to use for analysis
        
    Returns:
        List of detected UX issues
        
    Note:
        This requires an API key for GPT-4V, Claude, or similar.
        Set OPENAI_API_KEY environment variable or pass api_key.
    """
    import os
    import json
    import base64
    
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return [UXIssue(
            category=Category.FEEDBACK,
            severity=Severity.INFO,
            element="AI Analysis",
            description="No API key configured for AI analysis",
            suggestion="Set OPENAI_API_KEY environment variable"
        )]
    
    try:
        import openai
    except ImportError:
        return [UXIssue(
            category=Category.FEEDBACK,
            severity=Severity.INFO,
            element="AI Analysis",
            description="OpenAI package not installed",
            suggestion="pip install openai"
        )]
    
    # Encode image
    with open(screenshot_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    # Build checklist prompt
    checklist_text = "\n".join([
        f"- [{item.category.value}] {item.question}" 
        for item in UX_CHECKLIST
    ])
    
    prompt = f"""Analyze this UI screenshot for UX issues.

Check against this UX checklist:
{checklist_text}

For each issue found, respond in JSON format:
{{
    "issues": [
        {{
            "category": "usability|layout|feedback|controls|accessibility|consistency",
            "severity": "critical|major|minor|info",
            "element": "Name of the UI element",
            "description": "What's wrong",
            "suggestion": "How to fix it",
            "location": "Where in the UI (optional)"
        }}
    ]
}}

If no issues found, return {{"issues": []}}
"""
    
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"}
                    }
                ]
            }
        ],
        max_tokens=1500
    )
    
    # Parse response
    try:
        content = response.choices[0].message.content
        # Extract JSON from response
        import re
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            data = json.loads(json_match.group())
            issues = []
            for item in data.get("issues", []):
                issues.append(UXIssue(
                    category=Category(item.get("category", "usability")),
                    severity=Severity(item.get("severity", "minor")),
                    element=item.get("element", "Unknown"),
                    description=item.get("description", ""),
                    suggestion=item.get("suggestion", ""),
                    location=item.get("location")
                ))
            return issues
    except Exception as e:
        return [UXIssue(
            category=Category.FEEDBACK,
            severity=Severity.INFO,
            element="AI Analysis",
            description=f"Failed to parse AI response: {e}",
            suggestion="Check API response format"
        )]
    
    return []


def generate_issue_report(issues: List[UXIssue], title: str = "UX Analysis Report") -> str:
    """Generate markdown report of UX issues."""
    if not issues:
        return f"# {title}\n\n✅ No issues detected.\n"
    
    lines = [f"# {title}\n"]
    
    # Summary by severity
    by_severity = {}
    for issue in issues:
        by_severity.setdefault(issue.severity, []).append(issue)
    
    lines.append("## Summary\n")
    lines.append("| Severity | Count |")
    lines.append("|----------|-------|")
    for sev in [Severity.CRITICAL, Severity.MAJOR, Severity.MINOR, Severity.INFO]:
        count = len(by_severity.get(sev, []))
        icon = {"critical": "🔴", "major": "🟠", "minor": "🟡", "info": "🔵"}[sev.value]
        lines.append(f"| {icon} {sev.value.title()} | {count} |")
    
    lines.append("\n## Issues\n")
    
    for issue in sorted(issues, key=lambda x: list(Severity).index(x.severity)):
        icon = {"critical": "🔴", "major": "🟠", "minor": "🟡", "info": "🔵"}[issue.severity.value]
        lines.append(f"### {icon} [{issue.category.value.upper()}] {issue.element}\n")
        lines.append(f"**Problem:** {issue.description}\n")
        lines.append(f"**Suggestion:** {issue.suggestion}\n")
        if issue.location:
            lines.append(f"**Location:** {issue.location}\n")
        lines.append("")
    
    return "\n".join(lines)
