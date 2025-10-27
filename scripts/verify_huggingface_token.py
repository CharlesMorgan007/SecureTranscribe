#!/usr/bin/env python3
"""
Hugging Face Token Verification Script for SecureTranscribe
This script verifies that your HUGGINGFACE_TOKEN is working correctly
for accessing the PyAnnote speaker diarization model.
"""

import os
import sys
from pathlib import Path

# Add app directory to path
app_root = Path(__file__).parent.parent
sys.path.insert(0, str(app_root / "app"))


def verify_token():
    """Verify Hugging Face token works with PyAnnote model."""
    print("🔑 Verifying Hugging Face Token for PyAnnote Model")
    print("=" * 60)

    # Check if token is set
    token = os.environ.get("HUGGINGFACE_TOKEN")

    if not token:
        print("❌ HUGGINGFACE_TOKEN environment variable is not set")
        print("\n💡 To get a token:")
        print("   1. Go to: https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("   2. Accept the user conditions")
        print("   3. Go to: https://huggingface.co/settings/tokens")
        print("   4. Generate a new token and copy it")
        print("   5. Add to your .env file: HUGGINGFACE_TOKEN=hf_your_token")
        return False

    print(f"✅ Token found: {token[:10]}...{token[-4:]}")

    # Test token with PyAnnote
    try:
        from pyannote.audio import Pipeline

        print("\n🧪 Testing token with PyAnnote model...")

        # Attempt to load pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=token
        )

        print("✅ Successfully loaded PyAnnote pipeline!")
        print(f"   Model: pyannote/speaker-diarization-3.1")
        print(f"   Pipeline type: {type(pipeline)}")

        # Test basic functionality
        if hasattr(pipeline, "to"):
            print("✅ Pipeline supports device movement")

        print("\n🎉 Token is working correctly!")
        print("📚 Your diarization service will now use the real PyAnnote model")

        return True

    except Exception as e:
        print(f"❌ Token verification failed: {e}")

        # Provide specific guidance based on error
        error_str = str(e).lower()

        if "gated" in error_str or "private" in error_str:
            print("\n💡 The model is gated. You need to:")
            print(
                "   1. Go to: https://huggingface.co/pyannote/speaker-diarization-3.1"
            )
            print("   2. Accept the user conditions")
            print("   3. Your token should then work")

        elif "invalid" in error_str or "unauthorized" in error_str:
            print("\n💡 Token is invalid or expired:")
            print(
                "   1. Generate a new token at: https://huggingface.co/settings/tokens"
            )
            print("   2. Make sure you're copying the full token (starts with 'hf_')")

        elif "network" in error_str or "connection" in error_str:
            print("\n💡 Network connectivity issue:")
            print("   1. Check your internet connection")
            print("   2. Try again in a few moments")

        else:
            print("\n💡 General troubleshooting:")
            print("   1. Verify token is correctly copied (no extra spaces)")
            print("   2. Ensure token has correct permissions")
            print("   3. Try generating a new token")

        return False


def check_environment():
    """Check overall environment setup."""
    print("\n🌍 Environment Check:")
    print("-" * 30)

    env_vars = [
        ("HUGGINGFACE_TOKEN", "PyAnnote model access"),
        ("TEST_MODE", "Development mode"),
        ("MOCK_GPU", "GPU mocking"),
        ("CUDA_VISIBLE_DEVICES", "GPU availability"),
        ("LOG_LEVEL", "Logging level"),
    ]

    for var_name, description in env_vars:
        value = os.environ.get(var_name)
        if value:
            if var_name == "HUGGINGFACE_TOKEN":
                display_value = f"{value[:10]}...{value[-4:]}"
            else:
                display_value = value
            print(f"✅ {var_name}: {display_value} ({description})")
        else:
            print(f"⚠️  {var_name}: Not set ({description})")


def test_diarization_service():
    """Test diarization service with real model."""
    print("\n🧪 Testing Diarization Service with Real Model")
    print("-" * 50)

    try:
        from app.services.diarization_service import DiarizationService
        from app.models.transcription import Transcription

        # Initialize service
        service = DiarizationService()
        print("✅ DiarizationService initialized")

        # Load pipeline
        service._load_pipeline()

        if service.pipeline is None:
            print("❌ Pipeline is None after loading")
            return False

        # Check if it's using real pipeline (not mock)
        pipeline_type = type(service.pipeline).__name__
        print(f"✅ Pipeline loaded: {pipeline_type}")

        if "Mock" in pipeline_type:
            print("⚠️  Using mock pipeline (development mode)")
            print("💡 Set TEST_MODE=false and MOCK_GPU=false to use real model")
        else:
            print("🎉 Using real PyAnnote pipeline!")

        return True

    except Exception as e:
        print(f"❌ Diarization service test failed: {e}")
        return False


def show_next_steps():
    """Show what to do after verification."""
    print("\n📋 Next Steps:")
    print("-" * 20)
    print("1. ✅ Your token is verified and working")
    print("2. 🚀 Start your application:")
    print("   uvicorn app.main:app --reload --host 0.0.0.0 --port 8001")
    print("3. 🎤 Upload an audio file for transcription")
    print("4. 📊 Monitor real-time progress in the web interface")
    print("5. ✅ Your jobs should now complete successfully!")

    if os.environ.get("TEST_MODE") == "true":
        print("\n🔧 For production deployment:")
        print("   - Set TEST_MODE=false in your .env file")
        print("   - Set MOCK_GPU=false in your .env file")
        print("   - Ensure HUGGINGFACE_TOKEN is set in production environment")


if __name__ == "__main__":
    print("🤖 Hugging Face Token Verification for SecureTranscribe")
    print("This script ensures your token works with PyAnnote diarization model.")
    print()

    # Run verification
    token_valid = verify_token()

    if token_valid:
        check_environment()
        service_works = test_diarization_service()

        if service_works:
            show_next_steps()
            print(f"\n🎉 All checks passed! Your setup is ready.")
            sys.exit(0)

    print(f"\n⚠️  Some checks failed. Please resolve issues above.")
    sys.exit(1)
