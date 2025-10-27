#!/usr/bin/env python3
"""
Master Test Runner for SecureTranscribe
This script provides comprehensive testing for the transcription processing pipeline.
Run diagnostics, integration tests, and performance validation in one go.
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

# Add app directory to path
app_root = Path(__file__).parent.parent
sys.path.insert(0, str(app_root / "app"))


class TestRunner:
    """Master test runner for SecureTranscribe pipeline."""

    def __init__(self):
        self.app_root = app_root
        self.scripts_dir = self.app_root / "scripts"
        self.tests_dir = self.app_root / "tests"
        self.start_time = datetime.now()

    def print_banner(self):
        """Print test runner banner."""
        print("🚀 SecureTranscribe Master Test Runner")
        print("=" * 60)
        print(f"📅 Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📁 Project Root: {self.app_root}")
        print("=" * 60)

    def run_command(self, cmd: List[str], description: str) -> Tuple[bool, str]:
        """Run a command and return success status and output."""
        print(f"\n🧪 Running: {description}")
        print(f"💻 Command: {' '.join(cmd)}")
        print("-" * 40)

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.app_root),
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )

            # Print output in real-time
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)

            success = result.returncode == 0
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"\n{status}: {description}")

            return success, result.stdout + result.stderr

        except subprocess.TimeoutExpired:
            print(f"❌ TIMEOUT: {description} (exceeded 5 minutes)")
            return False, "Timeout exceeded"
        except Exception as e:
            print(f"❌ ERROR: {description} - {e}")
            return False, str(e)

    def run_diagnostics(self) -> bool:
        """Run pipeline diagnostics."""
        diagnostic_script = self.scripts_dir / "diagnose_pipeline.py"

        if not diagnostic_script.exists():
            print(f"❌ Diagnostic script not found: {diagnostic_script}")
            return False

        cmd = [sys.executable, str(diagnostic_script)]
        success, _ = self.run_command(cmd, "Pipeline Diagnostics")
        return success

    def run_integration_tests(self) -> bool:
        """Run integration tests."""
        test_script = self.scripts_dir / "run_pipeline_tests.py"

        if not test_script.exists():
            print(f"❌ Integration test script not found: {test_script}")
            return False

        cmd = [sys.executable, str(test_script)]
        success, _ = self.run_command(cmd, "Integration Tests")
        return success

    def run_unit_tests(self) -> bool:
        """Run unit tests using pytest."""
        unit_test_dir = self.tests_dir / "unit"

        if not unit_test_dir.exists():
            print(f"⚠️  Unit test directory not found: {unit_test_dir}")
            return True  # Skip if not present

        cmd = [
            sys.executable,
            "-m",
            "pytest",
            str(unit_test_dir),
            "-v",
            "--tb=short",
            "--color=yes",
        ]
        success, _ = self.run_command(cmd, "Unit Tests")
        return success

    def run_specific_test(self, test_path: str) -> bool:
        """Run a specific test file or test method."""
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            test_path,
            "-v",
            "--tb=short",
            "--color=yes",
        ]
        success, _ = self.run_command(cmd, f"Specific Test: {test_path}")
        return success

    def check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        print("\n🧪 Checking Dependencies...")

        required_packages = [
            "fastapi",
            "sqlalchemy",
            "pydantic",
            "pydantic-settings",
            "pytest",
            "uvicorn",
        ]

        missing_packages = []

        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package}")
            except ImportError:
                print(f"❌ {package} (missing)")
                missing_packages.append(package)

        if missing_packages:
            print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
            print("💡 Install with: pip install " + " ".join(missing_packages))
            return False

        print("✅ All required dependencies are available")
        return True

    def setup_test_environment(self) -> bool:
        """Set up test environment."""
        print("\n🧪 Setting up test environment...")

        # Set required environment variables for testing
        test_env = {
            "TEST_MODE": "true",
            "MOCK_GPU": "true",
            "CUDA_VISIBLE_DEVICES": "",
            "LOG_LEVEL": "DEBUG",
            "ALLOWED_HOSTS": '["localhost", "127.0.0.1"]',
            "CORS_ORIGINS": '["http://localhost:3000"]',
        }

        for key, value in test_env.items():
            os.environ[key] = value
            print(f"  📝 {key}={value}")

        # Create required directories
        required_dirs = [
            self.app_root / "uploads",
            self.app_root / "processed",
            self.app_root / "logs",
        ]

        for dir_path in required_dirs:
            dir_path.mkdir(exist_ok=True)
            print(f"  📁 Created: {dir_path}")

        print("✅ Test environment configured")
        return True

    def generate_report(self, results: dict) -> None:
        """Generate final test report."""
        end_time = datetime.now()
        duration = end_time - self.start_time

        print("\n" + "=" * 60)
        print("📊 FINAL TEST REPORT")
        print("=" * 60)
        print(f"📅 Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Duration: {duration.total_seconds():.2f} seconds")

        total_tests = len(results)
        passed_tests = sum(results.values())

        print(f"\n📈 Results: {passed_tests}/{total_tests} test suites passed")

        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status}: {test_name}")

        print(
            f"\n🎯 Overall Status: {'PASSED' if passed_tests == total_tests else 'FAILED'}"
        )

        if passed_tests == total_tests:
            print(
                "\n🎉 All tests passed! Your SecureTranscribe pipeline is working correctly."
            )
        else:
            failed_tests = [name for name, passed in results.items() if not passed]
            print(f"\n⚠️  Failed test suites: {', '.join(failed_tests)}")
            print("\n💡 Recommendations:")
            print("  1. Run the diagnostic script first to identify issues")
            print("  2. Check the logs for specific error messages")
            print("  3. Ensure all dependencies are installed")
            print("  4. Verify the application can start successfully")

    def run_all_tests(
        self, skip_diagnostics: bool = False, skip_unit: bool = False
    ) -> bool:
        """Run all test suites."""
        self.print_banner()

        results = {}

        # Check dependencies first
        if not self.check_dependencies():
            results["dependencies"] = False
            self.generate_report(results)
            return False

        results["dependencies"] = True

        # Set up test environment
        if not self.setup_test_environment():
            results["environment_setup"] = False
            self.generate_report(results)
            return False

        results["environment_setup"] = True

        # Run diagnostics (unless skipped)
        if not skip_diagnostics:
            results["diagnostics"] = self.run_diagnostics()
        else:
            print("\n⏭️  Skipping diagnostics (requested)")
            results["diagnostics"] = True

        # Run unit tests (unless skipped)
        if not skip_unit:
            results["unit_tests"] = self.run_unit_tests()
        else:
            print("\n⏭️  Skipping unit tests (requested)")
            results["unit_tests"] = True

        # Run integration tests
        results["integration_tests"] = self.run_integration_tests()

        # Generate final report
        self.generate_report(results)

        return all(results.values())

    def run_quick_test(self) -> bool:
        """Run a quick subset of tests for fast feedback."""
        self.print_banner()

        print("🏃 Running quick test suite...")

        results = {}

        # Basic checks
        results["dependencies"] = self.check_dependencies()
        results["environment_setup"] = self.setup_test_environment()

        if not all(results.values()):
            self.generate_report(results)
            return False

        # Quick diagnostic (just environment and imports)
        diagnostic_script = self.scripts_dir / "diagnose_pipeline.py"
        if diagnostic_script.exists():
            cmd = [sys.executable, str(diagnostic_script)]
            success, _ = self.run_command(cmd, "Quick Diagnostics")
            results["quick_diagnostics"] = success

        # Quick integration test
        test_script = self.scripts_dir / "run_pipeline_tests.py"
        if test_script.exists():
            cmd = [sys.executable, str(test_script)]
            success, _ = self.run_command(cmd, "Quick Integration Test")
            results["quick_integration"] = success

        self.generate_report(results)
        return all(results.values())


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SecureTranscribe Master Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_tests.py                    # Run all tests
  python scripts/run_tests.py --quick           # Quick test suite
  python scripts/run_tests.py --skip-diagnostics # Skip diagnostics
  python scripts/run_tests.py --specific tests/unit/test_api.py::test_health_check
        """,
    )

    parser.add_argument(
        "--quick", action="store_true", help="Run quick test suite for fast feedback"
    )

    parser.add_argument(
        "--skip-diagnostics", action="store_true", help="Skip diagnostic checks"
    )

    parser.add_argument("--skip-unit", action="store_true", help="Skip unit tests")

    parser.add_argument(
        "--specific", type=str, help="Run specific test file or test method"
    )

    parser.add_argument(
        "--diagnostics-only", action="store_true", help="Run only diagnostic checks"
    )

    args = parser.parse_args()

    runner = TestRunner()

    try:
        if args.specific:
            success = runner.run_specific_test(args.specific)
        elif args.diagnostics_only:
            runner.print_banner()
            success = runner.run_diagnostics()
        elif args.quick:
            success = runner.run_quick_test()
        else:
            success = runner.run_all_tests(
                skip_diagnostics=args.skip_diagnostics, skip_unit=args.skip_unit
            )

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test runner error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
