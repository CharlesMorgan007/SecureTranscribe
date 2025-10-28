#!/usr/bin/env python3
"""
Pipeline Diagnostics for SecureTranscribe
This script helps identify issues with the transcription processing pipeline.
"""

import os
import os
import sys
import json
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add app directory to path
app_root = Path(__file__).parent.parent
sys.path.insert(0, str(app_root))


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{'=' * 60}")
    print(f"🔍 {title}")
    print("=" * 60)


def print_subheader(title: str):
    """Print a formatted subheader."""
    print(f"\n--- {title} ---")


def check_environment():
    """Check environment variables and configuration."""
    print_header("Environment Check")

    # Check required environment variables
    required_vars = [
        "DATABASE_URL",
        "UPLOAD_DIR",
        "PROCESSED_DIR",
        "LOG_LEVEL",
        "ALLOWED_HOSTS",
        "CORS_ORIGINS",
    ]

    optional_vars = [
        "CUDA_VISIBLE_DEVICES",
        "WHISPER_MODEL_SIZE",
        "PYANNOTE_MODEL",
        "TEST_MODE",
        "MOCK_GPU",
    ]

    print("Required Environment Variables:")
    for var in required_vars:
        value = os.environ.get(var, "NOT_SET")
        status = "✅" if value != "NOT_SET" else "❌"
        print(f"  {status} {var}: {value}")

    print("\nOptional Environment Variables:")
    for var in optional_vars:
        value = os.environ.get(var, "NOT_SET")
        status = "✅" if value != "NOT_SET" else "⚠️"
        print(f"  {status} {var}: {value}")

    # Check directories
    print("\nDirectory Check:")
    dirs_to_check = [
        os.environ.get("UPLOAD_DIR", "./uploads"),
        os.environ.get("PROCESSED_DIR", "./processed"),
        "./logs",
    ]

    for dir_path in dirs_to_check:
        path = Path(dir_path)
        if path.exists():
            if path.is_dir():
                size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                file_count = len(list(path.rglob("*")))
                status = "✅"
                print(
                    f"  {status} {path} exists ({file_count} files, {size / 1024:.1f}KB)"
                )
            else:
                status = "❌"
                print(f"  {status} {path} exists but is not a directory")
        else:
            status = "❌"
            print(f"  {status} {path} does not exist")


def check_imports():
    """Check if all required modules can be imported."""
    print_header("Import Check")

    modules_to_test = [
        ("main", "FastAPI application"),
        ("app.core.config", "Configuration module"),
        ("app.core.database", "Database module"),
        ("app.models.processing_queue", "Processing queue model"),
        ("app.models.transcription", "Transcription model"),
        ("app.models.session", "Session model"),
        ("app.services.queue_service", "Queue service"),
        ("app.services.transcription_service", "Transcription service"),
        ("app.services.diarization_service", "Diarization service"),
        ("fastapi.testclient", "Test client"),
        ("sqlalchemy", "SQLAlchemy"),
    ]

    failed_imports = []

    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"✅ {module_name} - {description}")
        except ImportError as e:
            print(f"❌ {module_name} - {description}: {e}")
            failed_imports.append((module_name, str(e)))
        except Exception as e:
            print(f"⚠️  {module_name} - {description}: {e}")

    return len(failed_imports) == 0


def check_database():
    """Check database connectivity and state."""
    print_header("Database Check")

    try:
        from app.core.database import get_database
        from app.models.processing_queue import ProcessingQueue
        from app.models.transcription import Transcription
        from app.models.session import UserSession

        print("Testing database connection...")
        with next(get_database()) as db:
            # Test basic query
            result = db.execute("SELECT 1").fetchone()
            print(f"✅ Database connection successful: {result}")

            # Check tables
            print_subheader("Table Status")

            tables_to_check = [
                ("processing_queue", ProcessingQueue),
                ("transcriptions", Transcription),
                ("sessions", UserSession),
            ]

            for table_name, model_class in tables_to_check:
                try:
                    count = db.query(model_class).count()
                    print(f"✅ {table_name}: {count} records")
                except Exception as e:
                    print(f"❌ {table_name}: Error - {e}")

            # Check recent jobs
            print_subheader("Recent Jobs")
            try:
                recent_jobs = (
                    db.query(ProcessingQueue)
                    .order_by(ProcessingQueue.created_at.desc())
                    .limit(5)
                    .all()
                )

                if recent_jobs:
                    for job in recent_jobs:
                        job_dict = job.to_dict()
                        print(
                            f"  📋 Job {job_dict.get('job_id', 'Unknown')}: "
                            f"{job_dict.get('status', 'Unknown')} "
                            f"({job_dict.get('created_at', 'Unknown')})"
                        )
                else:
                    print("  ℹ️  No jobs found in queue")

            except Exception as e:
                print(f"❌ Error checking recent jobs: {e}")

        return True

    except Exception as e:
        print(f"❌ Database check failed: {e}")
        traceback.print_exc()
        return False


def check_queue_service():
    """Check queue service status and functionality."""
    print_header("Queue Service Check")

    try:
        from app.services.queue_service import get_queue_service

        print("Getting queue service instance...")
        queue_service = get_queue_service()

        if queue_service:
            print("✅ Queue service instantiated successfully")

            # Check service status
            status = queue_service.get_queue_status()
            print_subheader("Queue Status")
            print(f"  📊 Total jobs: {status.get('total_jobs', 0)}")
            print(f"  📊 Queued jobs: {status.get('queued_jobs', 0)}")
            print(f"  📊 Processing jobs: {status.get('processing_jobs', 0)}")
            print(f"  📊 Completed jobs: {status.get('completed_jobs', 0)}")
            print(f"  📊 Failed jobs: {status.get('failed_jobs', 0)}")

            # Check worker status
            try:
                worker_status = queue_service.get_worker_status()
                print_subheader("Worker Status")
                print(
                    f"  👥 Max workers: {worker_status.get('max_workers', 'Unknown')}"
                )
                print(
                    f"  👥 Active workers: {worker_status.get('active_workers', 'Unknown')}"
                )
                print(
                    f"  📝 Active jobs: {list(worker_status.get('active_jobs', {}).keys())}"
                )
            except Exception as e:
                print(f"⚠️  Worker status check failed: {e}")

            return True
        else:
            print("❌ Queue service is None")
            return False

    except Exception as e:
        print(f"❌ Queue service check failed: {e}")
        traceback.print_exc()
        return False


def check_api_endpoints():
    """Check API endpoints functionality."""
    print_header("API Endpoint Check")

    try:
        from main import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        endpoints_to_test = [
            ("GET", "/health", "Health check"),
            ("GET", "/info", "App info"),
            ("GET", "/", "Main page"),
            ("GET", "/api/sessions/create", "Session creation"),
            ("GET", "/api/queue/status", "Queue status"),
            ("GET", "/api/queue/worker-status", "Worker status"),
        ]

        working_endpoints = 0

        for method, path, description in endpoints_to_test:
            try:
                if method == "GET":
                    response = client.get(path)
                elif method == "POST":
                    response = client.post(path)

                if response.status_code == 200:
                    print(f"✅ {method} {path} - {description}")
                    working_endpoints += 1
                else:
                    print(
                        f"❌ {method} {path} - {description}: HTTP {response.status_code}"
                    )
                    try:
                        error_data = response.json()
                        print(f"    Error: {error_data.get('detail', 'Unknown error')}")
                    except:
                        print(f"    Response: {response.text[:100]}")

            except Exception as e:
                print(f"❌ {method} {path} - {description}: {e}")

        print(f"\n📊 {working_endpoints}/{len(endpoints_to_test)} endpoints working")
        return working_endpoints == len(endpoints_to_test)

    except Exception as e:
        print(f"❌ API endpoint check failed: {e}")
        traceback.print_exc()
        return False


def test_file_upload_workflow():
    """Test the complete file upload workflow."""
    print_header("File Upload Workflow Test")

    try:
        from main import app
        from fastapi.testclient import TestClient
        import tempfile
        import numpy as np
        from scipy.io import wavfile

        # Create test audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            # Generate simple sine wave
            sample_rate = 16000
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            frequency = 440
            amplitude = 0.3
            audio_data = (amplitude * np.sin(2 * np.pi * frequency * t) * 32767).astype(
                np.int16
            )
            wavfile.write(temp_file.name, sample_rate, audio_data)
            temp_path = temp_file.name

        try:
            client = TestClient(app)

            # Step 1: Create session
            print("Step 1: Creating session...")
            response = client.get("/api/sessions/create")
            if response.status_code != 200:
                print(f"❌ Session creation failed: {response.status_code}")
                return False

            session_data = response.json()
            session_id = session_data.get("session_id")
            print(f"✅ Session created: {session_id}")

            # Step 2: Upload file
            print("Step 2: Uploading file...")
            with open(temp_path, "rb") as f:
                response = client.post(
                    "/api/transcription/upload",
                    files={"file": ("test.wav", f, "audio/wav")},
                    data={"session_id": session_id},
                )

            if response.status_code != 200:
                print(f"❌ File upload failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False

            upload_data = response.json()
            transcription_id = upload_data.get("transcription_id")
            print(f"✅ File uploaded: {transcription_id}")

            # Step 3: Check upload status
            print("Step 3: Checking upload status...")
            response = client.get(f"/api/transcription/{transcription_id}/status")
            if response.status_code != 200:
                print(f"❌ Status check failed: {response.status_code}")
                return False

            status_data = response.json()
            current_status = status_data.get("status")
            print(f"✅ Current status: {current_status}")

            # Step 4: Check queue
            print("Step 4: Checking queue...")
            response = client.get("/api/queue/jobs")
            if response.status_code != 200:
                print(f"❌ Queue check failed: {response.status_code}")
                return False

            jobs_data = response.json()
            user_jobs = jobs_data.get("jobs", [])
            print(f"✅ User jobs in queue: {len(user_jobs)}")

            for job in user_jobs:
                print(f"  📋 Job {job.get('job_id')}: {job.get('status')}")

            return True

        finally:
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("  Install with: pip install scipy numpy")
        return False
    except Exception as e:
        print(f"❌ Upload workflow test failed: {e}")
        traceback.print_exc()
        return False


def check_gpu_services():
    """Check GPU service availability and configuration."""
    print_header("GPU Services Check")

    try:
+        from app.services.transcription_service import TranscriptionService
+        from app.services.diarization_service import DiarizationService

        print("Checking TranscriptionService...")
        trans_service = TranscriptionService()
        print(f"  📱 Model size: {trans_service.model_size}")
        print(f"  🔧 Device: {trans_service._get_device()}")
        print(f"  📦 Model loaded: {trans_service.model is not None}")

        print("\nChecking DiarizationService...")
        diar_service = DiarizationService()
        print(f"  🤖 Model: {diar_service.pyannote_model}")
        print(f"  🔧 Device: {diar_service._get_device()}")
        print(f"  📦 Pipeline loaded: {diar_service.pipeline is not None}")

        # Check environment
        print_subheader("GPU Environment")
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
        test_mode = os.environ.get("TEST_MODE", "Not set")
        mock_gpu = os.environ.get("MOCK_GPU", "Not set")

        print(f"  🎮 CUDA_VISIBLE_DEVICES: {cuda_devices}")
        print(f"  🧪 TEST_MODE: {test_mode}")
        print(f"  🎭 MOCK_GPU: {mock_gpu}")

        return True

    except Exception as e:
        print(f"❌ GPU services check failed: {e}")
        traceback.print_exc()
        return False


def check_logs():
    """Check recent log entries for errors."""
    print_header("Recent Logs Check")

    log_file = os.environ.get("LOG_FILE", "./logs/securetranscribe.log")
    log_path = Path(log_file)

    if not log_path.exists():
        print(f"⚠️  Log file not found: {log_path}")
        return False

    try:
        with open(log_path, "r") as f:
            lines = f.readlines()

        # Get last 50 lines
        recent_lines = lines[-50:] if len(lines) > 50 else lines

        print(f"📋 Showing last {len(recent_lines)} lines from {log_path}")

        error_count = 0
        warning_count = 0

        for line in recent_lines:
            line = line.strip()
            if "ERROR" in line:
                print(f"❌ {line}")
                error_count += 1
            elif "WARNING" in line:
                print(f"⚠️  {line}")
                warning_count += 1
            elif any(
                keyword in line
                for keyword in ["job", "processing", "queue", "transcription"]
            ):
                print(f"ℹ️  {line}")

        print_subheader("Summary")
        print(f"  ❌ Errors: {error_count}")
        print(f"  ⚠️  Warnings: {warning_count}")

        return error_count == 0

    except Exception as e:
        print(f"❌ Log check failed: {e}")
        return False


def run_comprehensive_check():
    """Run all diagnostic checks."""
    print("🔍 SecureTranscribe Pipeline Diagnostics")
    print(f"📅 Run at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🐍 Python version: {sys.version}")

    results = {}

    # Run all checks
+    results["environment"] = check_environment()
+    results["imports"] = check_imports()
+    results["database"] = check_database() if results["imports"] else None
+    results["queue_service"] = check_queue_service() if results["imports"] else None
+    results["api_endpoints"] = check_api_endpoints() if results["imports"] else None
+    results["file_upload"] = test_file_upload_workflow() if results["imports"] else None
+    results["gpu_services"] = check_gpu_services() if results["imports"] else None
+    results["logs"] = check_logs()

    # Print summary
    print_header("Diagnostic Summary")

    total_checks = len(results)
    passed_checks = sum(1 for v in results.values() if v is True)

    for check_name, passed in results.items():
        if passed is None:
            status = "⚠️  SKIP"
        elif passed:
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        print(f"{status}: {check_name.replace('_', ' ').title()}")

    print(f"\n📊 Overall: {passed_checks}/{total_checks} checks passed")

    if passed_checks == total_checks:
        print("🎉 All checks passed! Your pipeline should be working correctly.")
    else:
        print("⚠️  Some checks failed. Review the output above for details.")

        # Provide specific recommendations
        print_subheader("Recommendations")
        if not results["imports"]:
            print("💡 Install missing dependencies or check your virtual environment")
        if not results["database"]:
            print("💡 Check database file permissions and configuration")
        if not results["queue_service"]:
            print("💡 Queue service may not be running - check startup logs")
        if not results["api_endpoints"]:
            print("💡 Application may not be running - start with uvicorn")
        if not results["file_upload"]:
            print("💡 Check upload directory permissions and file size limits")
        if not results["gpu_services"]:
            print("💡 Install required ML models or enable MOCK_GPU mode")
        if not results["logs"]:
            print("💡 Review error logs for specific issues")

    return passed_checks == total_checks


if __name__ == "__main__":
    print("SecureTranscribe Pipeline Diagnostics")
    print(
        "This script helps identify issues with the transcription processing pipeline."
    )
    print()

    success = run_comprehensive_check()
    sys.exit(0 if success else 1)
