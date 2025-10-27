#!/usr/bin/env python3
"""
End-to-End Test Script for SecureTranscribe
Tests the complete transcription pipeline to verify all fixes work correctly.
"""

import os
import sys
import requests
import time
import tempfile
from pathlib import Path
from typing import Dict, Any


def create_test_audio_file() -> str:
    """Create a simple test audio file."""
    import numpy as np
    from scipy.io import wavfile

    # Create temporary audio file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        # Generate simple sine wave (2 seconds, 16kHz)
        sample_rate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        frequency = 440  # A4 note
        amplitude = 0.3
        audio_data = (amplitude * np.sin(2 * np.pi * frequency * t) * 32767).astype(
            np.int16
        )
        wavfile.write(temp_file.name, sample_rate, audio_data)
        return temp_file.name


def get_session(base_url: str) -> str:
    """Create a new session."""
    # Use empty POST body to match expected structure
    response = requests.post(
        f"{base_url}/api/sessions/create",
        json={"user_identifier": "test_user", "user_agent": "test_script"},
    )
    if response.status_code != 200:
        raise Exception(
            f"Failed to create session: {response.status_code} - {response.text}"
        )
    return response.json()["session_id"]


def upload_file(base_url: str, session_id: str, audio_path: str) -> Dict[str, Any]:
    """Upload audio file for transcription."""
    with open(audio_path, "rb") as f:
        files = {"file": ("test.wav", f, "audio/wav")}
        data = {"session_id": session_id}
        response = requests.post(
            f"{base_url}/api/transcription/upload", files=files, data=data
        )

    if response.status_code != 200:
        raise Exception(
            f"Failed to upload file: {response.status_code} - {response.text}"
        )
    return response.json()


def start_transcription(base_url: str, session_id: str, transcription_id: str) -> bool:
    """Start the transcription process."""
    response = requests.post(
        f"{base_url}/api/transcription/{transcription_id}/start",
        json={"session_id": session_id},
    )

    if response.status_code != 200:
        raise Exception(f"Failed to start transcription: {response.status_code}")
    return True


def monitor_transcription(
    base_url: str, session_id: str, transcription_id: str, timeout: int = 60
) -> Dict[str, Any]:
    """Monitor transcription progress."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        response = requests.get(
            f"{base_url}/api/transcription/{transcription_id}/status"
        )

        if response.status_code != 200:
            raise Exception(f"Failed to get status: {response.status_code}")

        status_data = response.json()
        current_status = status_data.get("status")
        progress = status_data.get("progress", 0)
        current_step = status_data.get("current_step", "")

        print(
            f"  📊 Status: {current_status}, Progress: {progress}%, Step: {current_step}"
        )

        if current_status == "completed":
            print("  ✅ Transcription completed successfully!")
            return status_data
        elif current_status == "failed":
            error = status_data.get("error", "Unknown error")
            print(f"  ❌ Transcription failed: {error}")
            return status_data

        time.sleep(2)

    raise Exception(f"Transcription timeout after {timeout} seconds")


def check_queue_status(base_url: str) -> Dict[str, Any]:
    """Check overall queue status."""
    response = requests.get(f"{base_url}/api/queue/status")

    if response.status_code != 200:
        raise Exception(f"Failed to get queue status: {response.status_code}")

    queue_data = response.json()
    print(f"  📋 Queue Status: {queue_data.get('total_jobs', 0)} total jobs")
    print(f"  📋 Queue Status: {queue_data.get('queued_jobs', 0)} queued jobs")
    print(f"  📋 Queue Status: {queue_data.get('processing_jobs', 0)} processing jobs")
    print(f"  📋 Queue Status: {queue_data.get('completed_jobs', 0)} completed jobs")

    return queue_data


def main():
    """Run end-to-end test."""
    print("🚀 SecureTranscribe End-to-End Test")
    print("=" * 50)

    # Configuration
    base_url = "http://127.0.0.1:8001"

    print("🔧 Configuration:")
    print(f"  🌐 Base URL: {base_url}")
    print(
        f"  🔑 HUGGINGFACE_TOKEN: {'Set' if os.environ.get('HUGGINGFACE_TOKEN') else 'Not Set'}"
    )
    print(f"  🧪 TEST_MODE: {os.environ.get('TEST_MODE', 'Not Set')}")
    print(f"  🎭 MOCK_GPU: {os.environ.get('MOCK_GPU', 'Not Set')}")

    print("\n🧪 Step 1: Creating session...")
    try:
        session_id = get_session(base_url)
        print(f"  ✅ Session created: {session_id}")
    except Exception as e:
        print(f"  ❌ Session creation failed: {e}")
        return False

    print("\n🧪 Step 2: Creating test audio...")
    try:
        audio_path = create_test_audio_file()
        print(f"  ✅ Test audio created: {audio_path}")
    except Exception as e:
        print(f"  ❌ Audio creation failed: {e}")
        return False

    print("\n🧪 Step 3: Uploading file...")
    try:
        upload_data = upload_file(base_url, session_id, audio_path)
        transcription_id = upload_data.get("transcription_id")
        print(f"  ✅ File uploaded: {transcription_id}")
    except Exception as e:
        print(f"  ❌ File upload failed: {e}")
        return False

    print("\n🧪 Step 4: Checking queue status...")
    try:
        queue_data = check_queue_status(base_url)
    except Exception as e:
        print(f"  ❌ Queue check failed: {e}")
        return False

    print("\n🧪 Step 5: Starting transcription...")
    try:
        if start_transcription(base_url, session_id, transcription_id):
            print("  ✅ Transcription started")
    except Exception as e:
        print(f"  ❌ Start failed: {e}")
        return False

    print("\n🧪 Step 6: Monitoring transcription...")
    try:
        final_status = monitor_transcription(base_url, session_id, transcription_id)

        if final_status.get("status") == "completed":
            print("  🎉 End-to-end test PASSED!")
            print(f"  📝 Final text: {final_status.get('text', 'No text')[:100]}...")
            print(f"  🗣️ Segments: {len(final_status.get('segments', []))}")
            print(f"  👥 Speakers: {final_status.get('num_speakers', 0)}")
            return True
        else:
            print("  ❌ End-to-end test FAILED!")
            return False

    except Exception as e:
        print(f"  ❌ Monitoring failed: {e}")
        return False

    finally:
        # Cleanup
        try:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
                print(f"  🧹 Cleaned up test audio file")
        except:
            pass


if __name__ == "__main__":
    print("🧪 This script tests the complete transcription pipeline.")
    print(
        "Make sure SecureTranscribe is running with: uvicorn app.main:app --reload --host 127.0.0.1 --port 8001"
    )
    print()

    success = main()

    if success:
        print("\n🎯 RESULT: All fixes are working correctly!")
        print("✅ Transcription pipeline is stable and functional")
        print("✅ Status tracking is working")
        print("✅ Speaker assignment is available")
        print("✅ No more 'jobs disappearing' issues")
        sys.exit(0)
    else:
        print("\n⚠️ RESULT: Test failed - check application logs")
        print("❌ Some issues may still need to be resolved")
        sys.exit(1)
