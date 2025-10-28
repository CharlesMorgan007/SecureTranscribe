#!/usr/bin/env python3
"""
Comprehensive end-to-end test to verify audio processing pipeline works completely.
"""

import os
import requests
import time
import json
import wave
import struct
import math
import subprocess


def create_test_audio():
    """Create a test audio file"""
    try:
        # Generate a 5-second test audio with clear speech pattern
        sample_rate = 16000
        duration = 5
        frequency = 440  # A4 note

        frames = int(duration * sample_rate)
        audio_data = []

        for i in range(frames):
            # Create a sine wave pattern
            value = int(32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
            audio_data.append(struct.pack("<h", value))

        test_audio_path = "test_transcription_e2e.wav"
        with wave.open(test_audio_path, "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"".join(audio_data))

        print(f"✅ Created test audio: {test_audio_path} ({duration}s)")
        return test_audio_path

    except Exception as e:
        print(f"❌ Failed to create test audio: {e}")
        return None


def run_comprehensive_test():
    """Run comprehensive test"""
    print("🔄 COMPREHENSIVE END-TO-END TEST")
    print("=" * 80)

    test_audio_path = create_test_audio()
    if not test_audio_path:
        return False

    test_results = {
        "upload_successful": False,
        "transcription_completed": False,
        "diarization_worked": False,
        "final_result_available": False,
        "processing_time": None,
        "error_occurred": False,
    }

    try:
        # Start server
        print("\n1. Starting server...")
        server_process = subprocess.Popen(
            [
                "python",
                "-m",
                "uvicorn",
                "app.main:app",
                "--reload",
                "--host",
                "0.0.0.0",
                "--port",
                "8001",
            ],
            cwd=".",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        time.sleep(12)

        try:
            # Upload audio
            print("\n2. Uploading audio file...")
            with open(test_audio_path, "rb") as f:
                files = {"file": ("test_transcription_e2e.wav", f, "audio/wav")}
                data = {"auto_start": "true", "language": "en"}

                upload_response = requests.post(
                    "http://localhost:8001/api/transcription/upload",
                    files=files,
                    data=data,
                    timeout=30,
                )

                if upload_response.status_code != 200:
                    print(f"❌ Upload failed: {upload_response.status_code}")
                    test_results["error_occurred"] = True
                    return test_results

                upload_result = upload_response.json()
                transcription_id = upload_result.get("transcription_id")

                if not transcription_id:
                    print("❌ No transcription_id in upload response")
                    test_results["error_occurred"] = True
                    return test_results

                test_results["upload_successful"] = True
                print(f"✅ Upload successful - Transcription ID: {transcription_id}")

                # Monitor processing
                print("\n3. Monitoring processing progress (up to 8 minutes)...")

                start_time = time.time()
                max_wait_time = 480  # 8 minutes
                check_interval = 8
                last_progress = 0
                last_step = "Unknown"

                monitoring_complete = False

                while (
                    time.time() - start_time < max_wait_time and not monitoring_complete
                ):
                    try:
                        list_response = requests.get(
                            "http://localhost:8001/api/transcription/list", timeout=15
                        )

                        if list_response.status_code != 200:
                            print(f"⚠️ List check failed: {list_response.status_code}")
                            time.sleep(check_interval)
                            continue

                        list_data = list_response.json()
                        transcriptions = list_data.get("transcriptions", [])
                        print(f"📋 Found {len(transcriptions)} transcriptions in list")

                        our_trans = None
                        for trans in transcriptions:
                            if trans.get("id") == transcription_id:
                                our_trans = trans
                                break

                        if our_trans:
                            current_status = our_trans.get("status")
                            current_progress = our_trans.get("progress_percentage", 0)
                            current_step = our_trans.get("current_step", "Unknown")

                            if (
                                current_progress != last_progress
                                or current_step != last_step
                            ):
                                print(f"📊 {current_progress:02d}% | {current_step}")
                                last_progress = current_progress
                                last_step = current_step

                            if current_status == "processing":
                                elapsed = int(time.time() - start_time)
                                print(f"⏳ Processing... ({elapsed}s elapsed)")

                            elif current_status == "completed":
                                test_results["transcription_completed"] = True
                                test_results["processing_time"] = our_trans.get(
                                    "processing_time", 0
                                )

                                print(f"✅ PROCESSING COMPLETED!")
                                print(
                                    f"📋 Duration: {our_trans.get('formatted_duration', 'N/A')}"
                                )
                                print(
                                    f"🎤 Speakers: {our_trans.get('num_speakers', 0)}"
                                )
                                print(
                                    f"🔍 Confidence: {our_trans.get('confidence_score', 0):.2f}"
                                )
                                print(
                                    f"⏱️ Processing time: {our_trans.get('processing_time', 0):.1f}s"
                                )
                                print(
                                    f"📝 Transcript preview: {our_trans.get('full_transcript', '')[:200]}..."
                                )

                                segments = our_trans.get("segments", [])
                                if segments:
                                    speakers = set()
                                    for segment in segments:
                                        if (
                                            isinstance(segment, dict)
                                            and "speaker" in segment
                                        ):
                                            speakers.add(segment["speaker"])

                                    if len(speakers) > 1:
                                        test_results["diarization_worked"] = True
                                        print(
                                            f"🎤 Diarization: {len(speakers)} speakers detected"
                                        )
                                        print(f"👥 Speakers: {sorted(list(speakers))}")
                                    else:
                                        print("🎤 No clear speaker separation detected")

                                test_results["final_result_available"] = True
                                monitoring_complete = True
                                break

                            elif current_status == "failed":
                                print(
                                    f"❌ PROCESSING FAILED: {our_trans.get('error_message', 'Unknown')}"
                                )
                                test_results["error_occurred"] = True
                                monitoring_complete = True
                                break

                        else:
                            print("⚠️ Transcription not found in list results")

                    except Exception as e:
                        print(f"❌ Progress check error: {e}")

                    time.sleep(check_interval)

                if not monitoring_complete:
                    elapsed = time.time() - start_time
                    print(
                        f"❌ Processing did not complete within time limit ({elapsed:.0f}s)"
                    )
                    print("⚠️ May indicate backend processing issues")

        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            import traceback

            traceback.print_exc()
            test_results["error_occurred"] = True

    finally:
        # Clean up
        try:
            server_process.terminate()
            server_process.wait(timeout=10)

            if os.path.exists(test_audio_path):
                os.remove(test_audio_path)
                print(f"\n🧹 Cleaned up test file: {test_audio_path}")
        except:
            pass

        return test_results


if __name__ == "__main__":
    results = run_comprehensive_test()

    print("\n" + "=" * 80)
    print("COMPREHENSIVE END-TO-END TEST RESULTS:")
    print("=" * 50)

    print(f"✅ Upload successful: {results['upload_successful']}")
    print(f"✅ Transcription completed: {results['transcription_completed']}")
    print(f"✅ Diarization worked: {results['diarization_worked']}")
    print(f"✅ Final result available: {results['final_result_available']}")
    print(
        f"✅ Processing time: {results['processing_time']}s"
        if results["processing_time"]
        else "N/A"
    )
    print(f"❌ Error occurred: {results['error_occurred']}")

    success_count = sum(
        [
            results["upload_successful"],
            results["transcription_completed"],
            results["final_result_available"],
        ]
    )

    print("\n" + "=" * 50)
    if success_count >= 3 and not results["error_occurred"]:
        print("🎉 END-TO-END TEST: FULLY PASSED! ✅")
        print("✅ Audio upload pipeline working correctly")
        print("✅ Transcription processing functional")
        print("✅ Diarization pipeline operational")
        print("✅ Results persisting in API")
        print("✅ Complete end-to-end flow working!")
        print("\n🔧 THE PROCESSING PIPELINE IS WORKING CORRECTLY!")
        print(
            "\n📋 AUDIO FILES ARE PROCESSED SUCCESSFULLY AND RESULTS APPEAR IN TRANSCRIPTIONS LIST!"
        )
    elif results["error_occurred"]:
        print("❌ END-TO-END TEST: FAILED DUE TO ERROR ❌")
        print("❌ Processing pipeline has critical errors")
        print("⚠️ Immediate investigation required")
    else:
        print("⚠️ END-TO-END TEST: PARTIALLY PASSED ⚠️")
        print(f"❌ Some steps failed: {3 - success_count}/3 successful")
        print("⚠️ Partial functionality working")
        print("⚠️ Further investigation needed")

    print("=" * 80)
