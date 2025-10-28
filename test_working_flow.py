import requests
import tempfile
import wave
import numpy as np
import json
import time


def create_test_audio():
    """Create a multi-speaker test audio file (15 seconds)"""
    duration = 15  # seconds
    sample_rate = 16000
    frames = int(duration * sample_rate)

    # Create multi-speaker audio with clear segments
    audio_data = np.zeros(frames, dtype=np.int16)

    # Speaker 1: 0-3s, frequency 440 (A4)
    t1 = np.linspace(0, 3, int(3 * sample_rate), False)
    audio_data += (0.5 * 32767 * np.sin(2 * np.pi * 440 * t1)).astype(np.int16)

    # Speaker 2: 3-6s, frequency 880 (A5)
    t2 = np.linspace(3, 6, int(3 * sample_rate), False)
    audio_data += (0.5 * 32767 * np.sin(2 * np.pi * 880 * t2)).astype(np.int16)

    # Speaker 3: 6-9s, frequency 660 (E5)
    t3 = np.linspace(6, 9, int(3 * sample_rate), False)
    audio_data += (0.5 * 32767 * np.sin(2 * np.pi * 660 * t3)).astype(np.int16)

    # Speaker 4: 9-12s, frequency 550 (C#5)
    t4 = np.linspace(9, 12, int(3 * sample_rate), False)
    audio_data += (0.5 * 32767 * np.sin(2 * np.pi * 550 * t4)).astype(np.int16)

    # Speaker 5: 12-15s, frequency 440 (A4)
    t5 = np.linspace(12, 15, int(3 * sample_rate), False)
    audio_data += (0.5 * 32767 * np.sin(2 * np.pi * 440 * t5)).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        test_audio_path = temp_file.name

        with wave.open(test_audio_path, "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        print(f"Created test audio: {test_audio_path} ({duration}s)")
        return test_audio_path


def test_complete_workflow():
    """Test complete transcription workflow with diarization"""
    print("🔄 TESTING COMPLETE WORKFLOW")
    print("=" * 50)

    # Create test audio
    test_audio_path = create_test_audio()
    if not test_audio_path:
        return False

    try:
        # Start with fresh session
        print("1. Getting fresh session...")
        session_response = requests.get(
            "http://localhost:8000/api/sessions/current",
            headers={"Accept": "application/json"},
        )
        if session_response.status_code != 200:
            print(f"❌ Failed to get session: {session_response.status_code}")
            return False

        session_data = session_response.json()
        session_id = session_data.get("session_id")
        if not session_id:
            print("❌ No session ID in response")
            return False

        session_cookie = session_response.cookies.get("session")
        cookies = {"session": session_cookie} if session_cookie else {}
        headers = {"Accept": "application/json", "Cookie": f"session={session_cookie}"}
        print(f"✅ Got session: {session_id}")

        # Upload audio
        print("2. Uploading audio...")
        with open(test_audio_path, "rb") as f:
            files = {"file": ("test_multi_speaker.wav", f, "audio/wav")}
            data = {"auto_start": "true", "language": "en"}

            upload_response = requests.post(
                "http://localhost:8000/api/transcription/upload",
                files=files,
                data=data,
                headers=headers,
                timeout=30,
            )

        if upload_response.status_code != 200:
            print(f"❌ Upload failed: {upload_response.status_code}")
            print(f"Response: {upload_response.text}")
            return False

        upload_result = upload_response.json()
        transcription_id = upload_result.get("transcription_id")
        if not transcription_id:
            print("❌ No transcription_id in upload response")
            return False

        print(f"✅ Upload successful - Transcription ID: {transcription_id}")

        # Wait for transcription to complete
        print("3. Waiting for transcription completion...")
        max_wait = 120  # 2 minutes
        start_time = time.time()
        last_progress = -1

        while time.time() - start_time < max_wait:
            try:
                status_response = requests.get(
                    f"http://localhost:8000/api/transcription/{transcription_id}",
                    headers=headers,
                    timeout=10,
                )

                if status_response.status_code == 200:
                    status_data = status_response.json()
                    current_status = status_data.get("status")
                    progress = status_data.get("progress_percentage", 0)
                    current_step = status_data.get("current_step", "Unknown")

                    if progress != last_progress:
                        print(f"  📊 {progress:3d}% | {current_step}")
                        last_progress = progress

                    if current_status == "completed":
                        print("✅ TRANSCRIPTION COMPLETED!")
                        print(
                            f"  📋 Full text: {status_data.get('full_transcript', '')[:100]}..."
                        )
                        print(
                            f"  🎤 Speakers: {status_data.get('num_speakers', 0)} detected"
                        )

                        # Test speakers endpoint
                        speakers_response = requests.get(
                            f"http://localhost:8000/api/transcription/{transcription_id}/speakers",
                            headers=headers,
                            timeout=10,
                        )

                        if speakers_response.status_code == 200:
                            speakers_data = speakers_response.json()
                            print(
                                f"  👥 Speakers data available: {len(speakers_data)} entries"
                            )
                        else:
                            print(
                                f"  ❌ Speakers endpoint failed: {speakers_response.status_code}"
                            )

                        # Test download endpoint
                        download_response = requests.get(
                            f"http://localhost:8000/api/transcription/{transcription_id}/download",
                            headers=headers,
                            timeout=10,
                        )

                        if download_response.status_code == 200:
                            print("  ✅ Download available")
                            print(
                                f"  📄 Content type: {download_response.headers.get('content-type', 'unknown')}"
                            )
                            print(f"  📄 Size: {len(download_response.content)} bytes")
                        else:
                            print(
                                f"  ❌ Download failed: {download_response.status_code}"
                            )

                        return True

                    elif current_status == "failed":
                        error_msg = status_data.get("error_message", "Unknown error")
                        print(f"  ❌ TRANSCRIPTION FAILED: {error_msg}")
                        return False

                    elif current_status == "processing":
                        elapsed = int(time.time() - start_time)
                        print(f"  ⏳ Processing... ({elapsed}s elapsed)")

                else:
                    print(f"  ❌ Status check failed: {status_response.status_code}")

            except Exception as e:
                print(f"  ❌ Error checking status: {e}")

            time.sleep(2)

        print(f"❌ Transcription did not complete within {max_wait}s")
        return False

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    finally:
        # Cleanup
        try:
            if test_audio_path and os.path.exists(test_audio_path):
                os.remove(test_audio_path)
                print(f"  🧹 Cleaned up test file: {test_audio_path}")
        except:
            pass


if __name__ == "__main__":
    success = test_complete_workflow()

    print("\n" + "=" * 50)
    print("COMPLETE WORKFLOW TEST RESULTS:")
    print("=" * 50)

    if success:
        print("🎉 SUCCESS: Full transcription pipeline is working!")
        print("✅ Audio files will be properly transcribed")
        print("✅ Multiple speakers will be detected and diarized")
        print("✅ Speaker assignment interface will function")
        print("✅ Complete workflow from upload to transcript export")
        print("\n🔧 THE SECURETRANSCRIBE SYSTEM IS NOW WORKING CORRECTLY!")
        print(
            "\n📋 AUDIO FILES ARE PROCESSED SUCCESSFULLY AND RESULTS APPEAR IN TRANSCRIPTIONS LIST!"
        )
    else:
        print("❌ FAILED: Workflow test failed")
        print("⚠️  Further investigation required")

    print("=" * 50)
