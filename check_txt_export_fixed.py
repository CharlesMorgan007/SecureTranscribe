#!/usr/bin/env python3
"""
Fixed TXT Export Check for SecureTranscribe
Properly checks the _export_txt method for common issues.
"""

import os
import re
from pathlib import Path


def check_txt_export():
    """Check the _export_txt method for issues."""
    print("🔍 Checking TXT Export Method...")
    print("=" * 50)

    # Read export service file
    export_file = Path("app/services/export_service.py")
    if not export_file.exists():
        print("❌ Export service file not found")
        return False

    content = export_file.read_text()

    # Find the line numbers for _export_txt method
    lines = content.split("\n")
    start_line = None
    end_line = None

    for i, line in enumerate(lines):
        if line.strip().startswith("def _export_txt("):
            start_line = i
        elif (
            start_line is not None
            and line.strip().startswith("def ")
            and not line.strip().startswith("def _export_txt")
        ):
            end_line = i
            break

    # If we didn't find an end, look for the next method or end of class
    if start_line is not None and end_line is None:
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if (
                line.strip()
                and not line.startswith("    ")
                and not line.startswith("\t")
            ):
                end_line = i
                break
        if end_line is None:
            end_line = len(lines)

    if start_line is None:
        print("❌ _export_txt method not found")
        return False

    # Extract method content (just the method body, excluding the def line)
    method_lines = (
        lines[start_line + 1 : end_line] if end_line else lines[start_line + 1 :]
    )
    method_body = "\n".join(method_lines)

    print("✅ _export_txt method found")

    # Check file-level imports
    print("📋 FILE-LEVEL IMPORTS:")
    file_checks = [
        ("import io", "import io"),
        ("import logging", "import logging"),
        ("ExportError import", "from app.utils.exceptions import ExportError"),
    ]

    file_issues = []
    for check_name, check_pattern in file_checks:
        if check_pattern in content:
            print(f"✅ {check_name}: Found")
        else:
            print(f"❌ {check_name}: Missing")
            file_issues.append(check_name)

    print("\n📋 METHOD-LEVEL CHECKS:")

    # Check method body for critical components
    method_checks = [
        ("StringIO initialization", "buffer = io.StringIO()"),
        ("Exception handling", "except Exception as e:"),
        ("UTF-8 encoding", '.encode("utf-8")'),
        ("Return statement", "return buffer.getvalue().encode("),
        ("Logger error call", 'logger.error(f"TXT export failed:'),
        ("ExportError raise", "raise ExportError("),
        ("Buffer seek", "buffer.seek(0)"),
    ]

    method_issues = []
    for check_name, check_pattern in method_checks:
        if check_pattern in method_body:
            print(f"✅ {check_name}: Found")
        else:
            print(f"❌ {check_name}: Missing")
            method_issues.append(check_name)

    # Additional structural checks
    print("\n📋 STRUCTURAL CHECKS:")

    # Check try-except structure
    if "try:" in method_body and "except" in method_body:
        print("✅ Try-except structure: Present")
    else:
        print("❌ Try-except structure: Missing")
        method_issues.append("Try-except structure")

    # Check method signature in full content
    if "def _export_txt(self, export_data: Dict[str, Any]) -> bytes:" in content:
        print("✅ Method signature: Correct")
    else:
        print("❌ Method signature: Incorrect")
        method_issues.append("Method signature")

    # Summary
    print("\n" + "=" * 50)
    print("📊 CHECK SUMMARY")
    print("=" * 50)
    total_checks = len(file_checks) + len(method_checks) + 2
    passed_checks = total_checks - len(file_issues) - len(method_issues)

    print(f"Total Checks: {total_checks}")
    print(f"Passed: {passed_checks}")
    print(f"Issues: {len(file_issues) + len(method_issues)}")

    if len(file_issues) + len(method_issues) == 0:
        print("🎉 TXT EXPORT METHOD: STRUCTURALLY SOUND")
        print("\n✅ All critical components are present!")
        print("If TXT export still fails:")
        print("1. Check production logs for specific error messages")
        print("2. Verify transcription data in database")
        print("3. Check for encoding issues with special characters")
        print("4. Check memory constraints for large transcriptions")
        return True
    else:
        print("⚠️ TXT EXPORT METHOD: HAS ISSUES")
        print("\nMissing components:")
        for issue in file_issues + method_issues:
            print(f"  - {issue}")
        if len(file_issues) > 0:
            print("\nNote: Import issues might be false positives if the")
            print("import statements have different formatting.")
        print("\nThese issues likely cause TXT export failures.")
        return False


def check_dependencies():
    """Check if TXT export dependencies are available."""
    print("\n🔍 Checking Dependencies...")

    try:
        import io

        print("✅ io module available")
    except ImportError:
        print("❌ io module missing")
        return False

    try:
        import json

        print("✅ json module available")
    except ImportError:
        print("❌ json module missing")
        return False

    return True


def main():
    """Run TXT export check."""
    print("🚀 SecureTranscribe TXT Export Check (FIXED)")
    print("=" * 60)

    # Check dependencies
    if not check_dependencies():
        print("\n💡 Install missing dependencies:")
        print("  pip install reportlab jinja2")
        return 1

    # Check TXT export method
    if check_txt_export():
        print("\n✅ TXT EXPORT METHOD ANALYSIS COMPLETE")
        return 0
    else:
        print("\n❌ TXT EXPORT METHOD ANALYSIS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
