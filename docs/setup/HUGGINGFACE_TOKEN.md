# Hugging Face Token Setup Guide for SecureTranscribe

This guide explains how to obtain and configure a Hugging Face token to access the PyAnnote speaker diarization model used by SecureTranscribe.

## 🔑 Why You Need a Token

The PyAnnote `speaker-diarization-3.1` model is a **gated model** on Hugging Face, meaning:
- You need to accept the user conditions
- You need a valid Hugging Face access token
- Without a token, diarization will fail and transcription jobs won't complete

## 🎯 Quick Setup Steps

### 1. Get Your Token

1. **Visit Hugging Face**: [https://huggingface.co](https://huggingface.co)
2. **Sign Up/Sign In** if you don't have an account
3. **Go to Model Page**: [https://huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
4. **Accept User Conditions** - You'll see a form asking you to accept the terms
5. **Generate Access Token**:
   - Click your profile → **Settings**
   - Go to **Access Tokens** (left sidebar)
   - Click **New token**
   - Name it (e.g., "SecureTranscribe")
   - Select **Read** permissions (minimum required)
   - Click **Generate token**
6. **Copy the token** (it starts with `hf_`)

### 2. Configure Your Token

#### Option A: .env File (Recommended)
Add to your `SecureTranscribe/.env` file:
```bash
# Hugging Face token for PyAnnote model access
HUGGINGFACE_TOKEN=hf_your_actual_token_here
```

#### Option B: Environment Variable (Temporary)
Set in your current shell session:
```bash
export HUGGINGFACE_TOKEN=hf_your_actual_token_here
```

#### Option C: Production Environment
Set in your production environment:
```bash
# In Dockerfile
ENV HUGGINGFACE_TOKEN=hf_your_production_token

# In systemd service
Environment=HUGGINGFACE_TOKEN=hf_your_production_token

# In Kubernetes
env:
- name: HUGGINGFACE_TOKEN
  value: "hf_your_production_token"
```

## ✅ Verify Your Token

Use the verification script to confirm everything works:
```bash
cd SecureTranscribe
source venv/bin/activate
python scripts/verify_huggingface_token.py
```

Expected output:
```
🔑 Verifying Hugging Face Token for PyAnnote Model
============================================================
✅ Token found: hf_xxxxxx...xxxx
🧪 Testing token with PyAnnote model...
✅ Successfully loaded PyAnnote pipeline!
   Model: pyannote/speaker-diarization-3.1
   Pipeline type: <class 'pyannote.audio.Pipeline'>
✅ Pipeline supports device movement

🎉 Token is working correctly!
📚 Your diarization service will now use the real PyAnnote model

🎉 All checks passed! Your setup is ready.
```

## 🔧 Configuration Options

### Development/Testing Environment
```bash
# For development - you can still use mock if token issues
TEST_MODE=false
MOCK_GPU=false
HUGGINGFACE_TOKEN=hf_your_dev_token
```

### Production Environment
```bash
# Production - ensure token is available
TEST_MODE=false
MOCK_GPU=false
HUGGINGFACE_TOKEN=hf_your_production_token
LOG_LEVEL=WARNING
```

### Fallback Behavior
SecureTranscribe will automatically fallback to mock diarization if:
- No token is provided
- Token is invalid/expired
- Model can't be loaded
- In test/mock mode

## 🚨 Important Security Notes

### Token Security
- **Never commit tokens to git** - add `.env` to `.gitignore`
- **Use read-only permissions** when possible
- **Rotate tokens regularly** - especially for production
- **Don't share tokens** - they're like passwords

### Best Practices
```bash
# .gitignore should include:
.env
.env.local
.env.production
*.key
*.pem
```

## 🔍 Troubleshooting

### Token Issues

#### "gated model" Error
```
Could not download 'pyannote/speaker-diarization-3.1' pipeline.
It might be because pipeline is private or gated...
```
**Solution**: 
1. Accept user conditions at the model page
2. Ensure token has proper permissions
3. Use the verification script

#### "invalid token" Error
```
401 Client Error: Unauthorized for url
```
**Solution**:
1. Generate a new token
2. Copy the full token (starts with `hf_`)
3. Ensure no extra spaces or line breaks

#### "permission denied" Error
```
Permission denied: token has insufficient scope
```
**Solution**:
1. Generate token with **Read** permissions
2. For production, consider **Write** permissions for model uploads

### Network Issues

#### Connection Timeout
```
TimeoutError: [Errno 11] Resource temporarily unavailable
```
**Solution**:
1. Check internet connection
2. Try again in a few moments
3. Consider using VPN if in restricted region

#### Firewall Issues
```
SSLError: [Errno 8] EOF occurred in violation of protocol
```
**Solution**:
1. Allow outbound HTTPS (port 443)
2. Check corporate firewall settings
3. Try from different network

## 📋 Verification Checklist

After setting up your token, verify:

- [ ] Token is accepted without errors
- [ ] PyAnnote pipeline loads successfully  
- [ ] Diarization works with real audio files
- [ ] Transcription jobs complete successfully
- [ ] Web interface shows correct progress
- [ ] No more "jobs disappearing" from queue

## 🔄 Token Management

### Multiple Environments
```bash
# Development
HUGGINGFACE_TOKEN=hf_dev_token_for_testing

# Staging  
HUGGINGFACE_TOKEN=hf_staging_token_for_testing

# Production
HUGGINGFACE_TOKEN=hf_production_token_for_real_use
```

### Token Rotation
```bash
# Generate new token every 90 days
# Update in .env file
# Restart application
# Verify with verification script
```

## 🎯 Next Steps

1. **Set your token** in the appropriate location
2. **Run verification**: `python scripts/verify_huggingface_token.py`
3. **Test the full pipeline**: `python scripts/run_pipeline_tests.py`
4. **Start your application**: `uvicorn app.main:app --reload`
5. **Upload audio file** and verify it completes successfully

## 📞 Getting Help

If you encounter issues:

1. **Check the verification script output** - it provides specific guidance
2. **Review the logs**: `tail -f logs/securetranscribe.log`
3. **Test with mock first**: Set `MOCK_GPU=true` to isolate issues
4. **Check Hugging Face status**: [https://status.huggingface.co](https://status.huggingface.co)

## 🔗 Useful Links

- **Model Page**: [https://huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- **Token Management**: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- **Hugging Face Docs**: [https://huggingface.co/docs/hub/security-tokens](https://huggingface.co/docs/hub/security-tokens)
- **PyAnnote Docs**: [https://github.com/pyannote/pyannote](https://github.com/pyannote/pyannote)

---

**Once your HUGGINGFACE_TOKEN is properly configured, your SecureTranscribe application will use the real PyAnnote speaker diarization model instead of falling back to mock services, providing much more accurate speaker identification and diarization results.**