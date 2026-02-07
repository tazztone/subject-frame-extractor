#!/usr/bin/env python
"""
Run comprehensive UX audit and generate report.

This script runs all UI/UX tests and generates a markdown report
with findings and recommendations.

Usage:
    python scripts/run_ux_audit.py                    # Run full audit
    python scripts/run_ux_audit.py --update-baselines # Update visual baselines
    python scripts/run_ux_audit.py --quick            # Quick component check only
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_tests(test_path: str, extra_args: list = None) -> tuple[int, str]:
    """Run pytest on specified test path and capture output."""
    cmd = [sys.executable, "-m", "pytest", test_path, "-v", "--tb=short"]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode, result.stdout + result.stderr


def generate_report(results: dict, output_path: Path) -> None:
    """Generate markdown report from test results."""
    report = [
        "# UX Audit Report",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n---\n",
        "## Summary\n",
    ]

    total_pass = sum(1 for r in results.values() if r["status"] == "pass")
    total_fail = sum(1 for r in results.values() if r["status"] == "fail")
    total_skip = sum(1 for r in results.values() if r["status"] == "skip")

    report.append("| Status | Count |")
    report.append("|--------|-------|")
    report.append(f"| ✅ Pass | {total_pass} |")
    report.append(f"| ❌ Fail | {total_fail} |")
    report.append(f"| ⏭️ Skip | {total_skip} |")

    report.append("\n---\n")
    report.append("## Test Results\n")

    for category, result in results.items():
        icon = {"pass": "✅", "fail": "❌", "skip": "⏭️"}.get(result["status"], "❓")
        report.append(f"### {icon} {category}\n")
        report.append(f"**Status**: {result['status'].upper()}")
        if result.get("output"):
            report.append(f"\n```\n{result['output'][-2000:]}\n```\n")

    report.append("\n---\n")
    report.append("## Recommendations\n")

    if total_fail > 0:
        report.append("1. Review failing tests and fix identified issues")
        report.append("2. Run `--update-baselines` after intentional UI changes")
        report.append("3. Check component verification for 'does nothing' bugs")
    else:
        report.append("✅ All tests passing! Continue regular testing schedule.")

    output_path.parent.mkdir(exist_ok=True, parents=True)
    output_path.write_text("\n".join(report))
    print(f"\n📄 Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run UX audit suite")
    parser.add_argument("--update-baselines", action="store_true", help="Update visual baselines")
    parser.add_argument("--quick", action="store_true", help="Quick component check only")
    parser.add_argument("--no-mock", action="store_true", help="Skip starting mock app (assume running)")
    args = parser.parse_args()

    print("🔍 Starting UX Audit...\n")

    results = {}
    project_root = Path(__file__).parent.parent

    # Start mock app if needed
    mock_process = None
    if not args.no_mock:
        print("🚀 Starting mock app...")
        mock_app_path = project_root / "tests" / "mock_app.py"
        mock_process = subprocess.Popen(
            [sys.executable, str(mock_app_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        import time

        time.sleep(8)  # Wait for Gradio to start

    try:
        # 1. Component Verification
        print("\n📦 Running Component Verification...")
        code, output = run_tests("tests/ui/test_component_verification.py")
        results["Component Verification"] = {
            "status": "pass" if code == 0 else ("fail" if "FAILED" in output else "skip"),
            "output": output,
        }

        if args.quick:
            print("⏩ Quick mode - skipping visual regression")
        else:
            # 2. Visual Regression
            print("\n🖼️ Running Visual Regression Tests...")
            extra_args = ["--update-baselines"] if args.update_baselines else []
            code, output = run_tests("tests/ui/test_visual_regression.py", extra_args)
            results["Visual Regression"] = {
                "status": "pass" if code == 0 else ("fail" if "FAILED" in output else "skip"),
                "output": output,
            }

        # 3. Main Flow Tests
        print("\n🔄 Running Main Flow Tests...")
        code, output = run_tests("tests/ui/test_app_flow.py")
        results["Main App Flow"] = {
            "status": "pass" if code == 0 else ("fail" if "FAILED" in output else "skip"),
            "output": output,
        }

        # Generate report
        report_dir = project_root / "ux_reports"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"ux_audit_{timestamp}.md"
        generate_report(results, report_path)

        # Print summary
        print("\n" + "=" * 50)
        print("📊 UX AUDIT SUMMARY")
        print("=" * 50)
        for category, result in results.items():
            icon = {"pass": "✅", "fail": "❌", "skip": "⏭️"}.get(result["status"], "❓")
            print(f"  {icon} {category}: {result['status'].upper()}")

    finally:
        if mock_process:
            print("\n🛑 Stopping mock app...")
            mock_process.terminate()
            mock_process.wait(timeout=5)

    print("\n✨ UX Audit complete!")


if __name__ == "__main__":
    main()
