#!/usr/bin/env python3
"""
TXT Export Analysis Script for SecureTranscribe
Analyzes the _export_txt method to identify potential issues.
"""

import os
import re
from pathlib import Path


def analyze_txt_export_method():
    """Analyze the _export_txt method for potential issues."""
    print("🔍 Analyzing TXT Export Method...")
    print("=" * 60)

    # Read the export service file
    export_file = Path("app/services/export_service.py")
    if not export_file.exists():
        print("❌ Export service file not found")
        return False

    content = export_file.read_text()

    # Find the _export_txt method
    export_txt_match = re.search(
        r"def _export_txt\(self.*?\):(.*?)(?=\ndef|\Z)", content, re.DOTALL
    )

    if not export_txt_match:
        print("❌ _export_txt method not found")
        return False

    export_txt_body = export_txt_match.group(1)
    print("✅ Found _export_txt method")

    # Analyze for potential issues
    issues_found = []

    # Check 1: String formatting issues
    if '"created_at"][:10]' in export_txt_body:
        issues_found.append('Potential string slicing issue: created_at"][:10"]')

    # Check 2: Missing error handling
    if "except Exception as e:" not in export_txt_body:
        issues_found.append("Missing exception handling in _export_txt")

    # Check 3: Encoding issues
    if '.encode("utf-8")' not in export_txt_body:
        issues_found.append("Missing UTF-8 encoding")

    # Check 4: Buffer handling
    if "buffer = io.StringIO()" not in export_txt_body:
        issues_found.append("Missing StringIO buffer initialization")

    # Check 5: Return statement
    if 'return buffer.getvalue().encode("utf-8")' not in export_txt_body:
        issues_found.append("Missing or incorrect return statement")

    # Check 6: Method calls that might fail
    risky_patterns = [
        r'get\([\'"][^\'"]+[\'"]\)',
        r"\.format\(",
        r"%[^s]*\(",
    ]

    for pattern, description in risky_patterns:
        if re.search(pattern, export_txt_body):
            issues_found.append(f"Risky pattern: {description}")

    # Check 7: Missing imports
    if "import io" not in content[:500] and "StringIO" in export_txt_body:
        issues_found.append("StringIO used but io not imported")

    print(f"\n📊 Analysis Results:")
    print(f"Method length: {len(export_txt_body)} characters")

    if issues_found:
        print(f"\n❌ Potential Issues Found ({len(issues_found)}):")
        for i, issue in enumerate(issues_found, 1):
            print(f"  {i}. {issue}")
    else:
        print("✅ No obvious issues found in _export_txt method")

    return len(issues_found) == 0


def check_file_permissions():
    """Check file permissions for temp directories."""
    print("🔍 Checking File Permissions...")

    import tempfile
    import stat

    # Check temp directory
    try:
        temp_dir = tempfile.gettempdir()
        test_file = os.path.join(temp_dir, "securetranscribe_test.txt")

        # Test write permission
        with open(test_file, "w") as f:
            f.write("test")

        # Test read permission
        with open(test_file, "r") as f:
            content = f.read()

        # Clean up
        os.remove(test_file)

        print("✅ Temp directory permissions: OK")
        return True

    except PermissionError as e:
        print(f"❌ Permission error in temp directory: {e}")
        return False
    except Exception as e:
        print(f"❌ Error checking permissions: {e}")
        return False


def check_common_txt_export_issues():
    """Check for common TXT export issues."""
    print("🔍 Checking Common TXT Export Issues...")

    issues = []

    # Check temp directory permissions
    if not check_file_permissions():
        issues.append("Temp directory permission error")

    # Check encoding support
    try:
        test_string = "Test with special chars: àáéíóú"
        encoded = test_string.encode("utf-8")
        decoded = encoded.decode("utf-8")
        if test_string == decoded:
            print("✅ UTF-8 encoding: OK")
        else:
            issues.append("UTF-8 encoding issue detected")
    except Exception as e:
        issues.append(f"Encoding test error: {e}")

    # Check StringIO availability
    try:
        import io

        buffer = io.StringIO()
        buffer.write("test")
        print("✅ StringIO: OK")
    except Exception as e:
        issues.append(f"StringIO error: {e}")

    return issues


def main():
    """Run TXT export analysis and provide recommendations."""
    print("🚀 SecureTranscribe TXT Export Analysis")
    print("=" * 60)

    # Analyze the export method
    method_ok = analyze_txt_export_method()

    # Check for common issues
    common_issues = check_common_txt_export_issues()

    # Overall assessment
    print("\n" + "=" * 60)
    print("📋 OVERALL ASSESSMENT")
    print("=" * 60)

    if method_ok and len(common_issues) == 0:
        print("✅ TXT EXPORT METHOD ANALYSIS: PASSED")
        print("\n🔧 If TXT export still fails in production:")
        print("  1. Check production logs for specific error messages:")
        print("     tail -f logs/securetranscribe.log | grep -i 'txt'")
        print("  2. Verify transcription data structure in database:")
        print(
            "     SELECT id, full_transcript FROM transcriptions WHERE id = <transcription_id>"
        )
        print("  3. Test with different transcription content:")
        print("     - Simple text without special characters")
        print("     - Different speaker assignments")
        print("  4. Check for large transcription memory issues:")
        print("     - Monitor memory usage during export")
        print("  5. Verify export dependencies:")
        print("     pip show reportlab jinja2")
        return 0
    else:
        print("❌ TXT EXPORT METHOD ANALYSIS: ISSUES FOUND")

        if method_ok:
            print("\n🔧 TXT Export Method Fixes:")
            print("  The _export_txt method has structural issues that should be fixed")

        if common_issues:
            print("\n⚠️ Environment Issues:")
            for issue in common_issues:
                print(f"  - {issue}")

        print("\n🛠️ Recommended Actions:")
        print("  1. Run: python3 analyze_txt_export.py")
        print("  2. Apply fixes based on analysis results")
        print("  3. Test export in isolated environment")
        print("  4. Check production server environment")

        return 1


if __name__ == "__main__":
    exit(main())
