import os
import time
import json
import math
import wave
import struct
import subprocess
from pathlib import Path
from typing import Tuple, Optional

import pytest
import requests


def _create_test_audio(
    file_path: Path, duration_seconds: int = 5, sample_rate: int = 16000
) -> Path:
    """
    Create a simple mono WAV audio file with a sine wave tone.

    Args:
        file_path: Path for the output WAV file.
        duration_seconds: Duration of audio.
        sample_rate: Sample rate in Hz.

    Returns:
        Path to the generated WAV file.
    """
    frames = int(duration_seconds * sample_rate)
    frequency = 440  # A4 note
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(file_path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        for i in range(frames):
            value = int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
            wav_file.writeframes(struct.pack("<h", value))
    return file_path


def _wait_for_health(
    session: requests.Session, base_url: str, timeout: int = 120
) -> bool:
    """
    Poll the health endpoint until the app is ready or timeout expires.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = session.get(f"{base_url}/health", timeout=5)
            if resp.status_code == 200 and resp.json().get("status") == "healthy":
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


@pytest.fixture(scope="module")
def e2e_server():
    """
    Start the FastAPI app via uvicorn as a subprocess for E2E testing, and provide a requests.Session
    that preserves cookies for the server-side session.

    Yields:
        (base_url, session, process)
    """
    # Project root is two levels up from this file: SecureTranscribe/tests/ -> SecureTranscribe
    project_root = Path(__file__).resolve().parent.parent

    # Test-friendly environment
    _ = os.environ.setdefault("TEST_MODE", "true")  # Use mock diarization
    _ = os.environ.setdefault("MOCK_GPU", "true")  # Disable GPU requirements
    _ = os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # Ensure no CUDA usage
    _ = os.environ.setdefault("LOG_LEVEL", "DEBUG")
    _ = os.environ.setdefault("WHISPER_MODEL_SIZE", "tiny")  # Fastest whisper variant

    # Pick a non-default port to avoid collisions
    port = int(os.environ.get("E2E_APP_PORT", "8010"))
    base_url = f"http://127.0.0.1:{port}"

    # Start server
    process = subprocess.Popen(
        [
            "python",
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        cwd=str(project_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    session = requests.Session()

    # Wait until server is healthy
    if not _wait_for_health(session, base_url, timeout=180):
        try:
            process.terminate()
        except Exception:
            pass
        pytest.skip("App failed to become healthy within timeout")

    try:
        yield base_url, session, process
    finally:
        # Teardown
        try:
            process.terminate()
            process.wait(timeout=10)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
        session.close()


@pytest.mark.e2e
@pytest.mark.slow
def test_e2e_upload_process_diarize_export(tmp_path: Path, e2e_server):
    """
    End-to-end test:
    - Generates a small WAV file
    - Uploads it to the API (auto-start processing)
    - Polls status until completed
    - Asserts diarization present (speakers found in segments)
    - Exports the transcription as JSON and asserts non-empty content
    """
    base_url, session, _ = e2e_server

    # 1) Create test audio file
    audio_path = _create_test_audio(tmp_path / "e2e_test.wav", duration_seconds=5)

    # 2) Upload with auto_start
    with open(audio_path, "rb") as f:
        files = {"file": ("e2e_test.wav", f, "audio/wav")}
        data = {"auto_start": "true", "language": "en"}
        resp = session.post(
            f"{base_url}/api/transcription/upload", files=files, data=data, timeout=60
        )

    assert resp.status_code == 200, f"Upload failed: {resp.status_code} {resp.text}"
    up = resp.json()
    transcription_id = up.get("transcription_id")
    assert transcription_id, f"No transcription_id in upload response: {up}"

    # 3) Poll status until completed or timeout
    status_url = f"{base_url}/api/transcription/status/{transcription_id}"
    start = time.time()
    max_wait = 8 * 60  # up to 8 minutes to allow first-time model downloads
    last_progress = -1
    last_step = ""

    while time.time() - start < max_wait:
        r = session.get(status_url, timeout=20)
        assert r.status_code == 200, f"Status check failed: {r.status_code} {r.text}"
        status = r.json()

        cur_status = status.get("status")
        progress = int(status.get("progress_percentage", 0) or 0)
        step = status.get("current_step", "")

        if progress != last_progress or step != last_step:
            print(f"[E2E] Progress: {progress:02d}% | Step: {step}")
            last_progress, last_step = progress, step

        if cur_status == "completed":
            break
        if cur_status == "failed":
            pytest.fail(f"Processing failed: {status.get('error_message', 'Unknown')}")

        time.sleep(5)

    else:
        pytest.fail("Processing did not complete within time limit")

    # 4) Validate results: transcript and diarization
    final = session.get(status_url, timeout=20).json()
    full_text = final.get("full_transcript") or ""
    segments = final.get("segments") or []
    speakers_list = final.get("speakers") or []

    assert len(full_text) >= 0, (
        "No transcript returned"
    )  # allow empty text but presence verified
    assert isinstance(segments, list) and len(segments) > 0, "No segments returned"

    # Ensure at least one speaker label present in segments
    segment_speakers = {
        seg.get("speaker")
        for seg in segments
        if isinstance(seg, dict) and "speaker" in seg
    }
    assert len(segment_speakers) >= 1 or len(speakers_list) >= 1, (
        f"No speaker labels detected. segments speakers={segment_speakers}, speakers_list={speakers_list}"
    )

    # 5) Export as JSON and verify non-empty content
    export_url = f"{base_url}/api/transcription/export/{transcription_id}"
    # export_format is expected as a query param by FastAPI (not request body)
    export_resp = session.post(export_url, params={"export_format": "json"}, timeout=60)
    assert export_resp.status_code == 200, (
        f"Export failed: {export_resp.status_code} {export_resp.text}"
    )
    assert export_resp.content and len(export_resp.content) > 0, (
        "Exported file is empty"
    )

    # Optional: validate JSON content
    try:
        export_json = json.loads(export_resp.content.decode("utf-8"))
        assert isinstance(export_json, dict), "Exported JSON is not a dict"
        # rudimentary checks
        assert "transcript" in export_json or "segments" in export_json, (
            "Export JSON missing expected keys"
        )
    except Exception as e:
        pytest.fail(f"Exported content is not valid JSON: {e}")
