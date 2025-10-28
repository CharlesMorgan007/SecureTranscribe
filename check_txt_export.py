#!/usr/bin/env python3
"""
Simple TXT Export Check for SecureTranscribe
Checks the _export_txt method for common issues.
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

    # Find _export_txt method
    export_txt_match = re.search(r'def _export_txt\(self.*?\):(.*?)(?=\ndef|\Z)', content, re.DOTALL)

    if not export_txt_match:
        print("❌ _export_txt method not found")
        return False

    print("✅ _export_txt method found")

    # Check method body for critical components
    method_body = export_txt_match.group(1)

    # Critical checks
    checks = [
        ("StringIO initialization", "buffer = io.StringIO()"),
        ("Exception handling", "except Exception as e:"),
        ("UTF-8 encoding", ".encode('utf-8')"),
        ("Return statement", "return buffer.getvalue().encode('utf-8')"),
        ("Import statement", "import io"),
    ]

    issues_found = []
    for check_name, check_pattern in checks:
        if check_pattern in method_body:
            print(f"✅ {check_name}: Found")
        else:
            print(f"❌ {check_name}: Missing")
            issues_found.append(check_name)

    # Summary
    print("\n" + "=" * 50)
    print("📊 CHECK SUMMARY")
    print("=" * 50)
    print(f"Total Checks: {len(checks)}")
    print(f"Passed: {len(checks) - len(issues_found)}")
    print(f"Issues: {len(issues_found)}")

    if len(issues_found) == 0:
        print("🎉 TXT EXPORT METHOD: STRUCTURALLY SOUND")
        print("\nIf TXT export still fails:")
        print("1. Check production logs for specific error messages")
        print("2. Verify transcription data in database")
        print("3. Check for encoding issues with special characters")
        print("4. Check memory constraints for large transcriptions")
        return True
    else:
        print("⚠️ TXT EXPORT METHOD: HAS ISSUES")
        print("\nMissing components:")
        for issue in issues_found:
            print(f"  - {issue}")
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
    print("🚀 SecureTranscribe TXT Export Check")
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
```

Now let me run this simple check:
<tool_call>terminal
<arg_key>command</arg_key>
<arg_value>python3 check_txt_export.py</arg_value>
<arg_key>cd</arg_key>
<arg_value>/Users/cmorgan/Devel/Personal/SecureTranscribe</arg_value>
</tool_call>
