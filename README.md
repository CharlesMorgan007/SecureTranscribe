# SecureTranscribe

A secure, offline audio transcription and speaker diarization application with GPU acceleration support.

## Overview

SecureTranscribe is a Python web application that provides secure, offline audio transcription and speaker diarization capabilities. It processes audio files locally, ensuring complete privacy and confidentiality of your data while leveraging NVIDIA GPU acceleration for optimal performance.

## Features

- **Secure Offline Processing**: All audio processing happens locally on your machine
- **GPU Acceleration**: Leverages NVIDIA RTX 4090 for faster processing
- **Speaker Diarization**: Automatically identifies and separates different speakers
- **Web Interface**: User-friendly web-based interface for audio upload and management
- **Speaker Recognition**: Stores speaker traits for automatic identification in future sessions
- **Multiple Export Formats**: PDF, CSV, TXT, JSON export options
- **Session Management**: Supports multiple users with queue-based processing
- **Real-time Progress Tracking**: Visual feedback during processing
- **Comprehensive Documentation**: Full setup guides for local and cloud deployment

## Requirements

- Python 3.11+
- NVIDIA GPU (RTX 4090 recommended for optimal performance)
- CUDA Toolkit 11.8+ (for GPU acceleration)
- 16GB+ RAM recommended
- 10GB+ free disk space

## Installation

### Local Development Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/SecureTranscribe.git
   cd SecureTranscribe
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download Required Models**
   ```bash
   python -m spacy download en_core_web_sm
   ```

5. **Initialize Database**
   ```bash
   python -m app.core.database init
   ```

6. **Run the Application**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Docker Deployment

1. **Build Docker Image**
   ```bash
   docker build -t securetranscribe .
   ```

2. **Run Container**
   ```bash
   docker run -p 8000:8000 --gpus all -v $(pwd)/uploads:/app/uploads securetranscribe
   ```

## Cloud Deployment

### AWS EC2 Setup

1. **Launch EC2 Instance**
   - Choose GPU instance (g4dn.xlarge or larger)
   - Use Ubuntu 22.04 LTS AMI
   - Configure security groups for ports 80, 443, 8000

2. **Install Dependencies**
   ```bash
   sudo apt update
   sudo apt install python3.11 python3.11-venv python3-pip
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-11-8
   ```

3. **Deploy Application**
   ```bash
   git clone https://github.com/yourusername/SecureTranscribe.git
   cd SecureTranscribe
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

### Google Cloud Platform

1. **Create GPU VM Instance**
   ```bash
   gcloud compute instances create securetranscribe-vm \
     --machine-type=n1-standard-4 \
     --accelerator=type=nvidia-tesla-t4,count=1 \
     --image-family=ubuntu-2204-lts \
     --image-project=ubuntu-os-cloud \
     --boot-disk-size=100GB
   ```

2. **Install NVIDIA Drivers and Application**
   ```bash
   gcloud ssh securetranscribe-vm
   # Follow same installation steps as AWS
   ```

## Usage

### Web Interface

1. **Access the Application**
   - Open browser to `http://localhost:8000`

2. **Upload Audio File**
   - Click "Choose File" and select your audio file
   - Supported formats: MP3, WAV, M4A, FLAC, OGG

3. **Process Audio**
   - Click "Start Transcription"
   - Monitor progress in real-time
   - Wait for speaker identification phase

4. **Label Speakers**
   - Listen to 2-10 second clips of each speaker
   - Assign names to identified speakers

5. **Export Results**
   - Choose export format (PDF, CSV, TXT, JSON)
   - Select additional content options
   - Download final transcript

### API Usage

```python
import requests

# Upload audio file
files = {'file': open('audio.mp3', 'rb')}
response = requests.post('http://localhost:8000/api/upload', files=files)

# Start processing
job_id = response.json()['job_id']
response = requests.post(f'http://localhost:8000/api/process/{job_id}')

# Check status
response = requests.get(f'http://localhost:8000/api/status/{job_id}')
status = response.json()['status']

# Download results
response = requests.get(f'http://localhost:8000/api/download/{job_id}')
with open('transcript.pdf', 'wb') as f:
    f.write(response.content)
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Database
DATABASE_URL=sqlite:///./securetranscribe.db

# Application
SECRET_KEY=your-secret-key-here
DEBUG=False
HOST=0.0.0.0
PORT=8000

# GPU Settings
CUDA_VISIBLE_DEVICES=0
TORCH_CUDA_ARCH_LIST="8.6"  # For RTX 4090

# File Paths
UPLOAD_DIR=./uploads
PROCESSED_DIR=./processed
MAX_FILE_SIZE=500MB

# Processing
MAX_WORKERS=4
QUEUE_SIZE=10
CLEANUP_DELAY=3600  # 1 hour in seconds
```

### Audio Processing Settings

```python
# In app/core/config.py
AUDIO_SETTINGS = {
    'sample_rate': 16000,
    'chunk_length_s': 30,
    'overlap_length_s': 5,
    'max_speakers': 10,
    'min_speaker_duration': 2.0,
    'confidence_threshold': 0.8
}
```

## Architecture

```
SecureTranscribe/
├── app/
│   ├── api/              # API endpoints
│   ├── core/             # Core configuration and database
│   ├── models/           # Database models
│   ├── services/         # Business logic
│   ├── static/           # CSS, JS, images
│   ├── templates/        # HTML templates
│   └── utils/            # Utility functions
├── tests/                # Test suite
├── docs/                 # Documentation
├── uploads/              # Temporary upload storage
├── processed/            # Processed files storage
└── requirements.txt      # Python dependencies
```

## Development

### Running Tests

```bash
# Run all tests with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/unit/test_audio_processing.py

# Run with verbose output
pytest -v tests/
```

### Code Quality

```bash
# Format code
black app/ tests/

# Lint code
flake8 app/ tests/

# Type checking
mypy app/

# Run all quality checks
pre-commit run --all-files
```

### Adding New Features

1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement changes with tests
3. Ensure 80%+ test coverage
4. Run quality checks
5. Submit pull request

## Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size in configuration
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

**Audio Format Not Supported**
```bash
# Install additional codecs
sudo apt install ffmpeg libsndfile1
```

**GPU Not Detected**
```bash
# Check CUDA installation
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Performance Optimization

1. **Use Faster Models**
   ```python
   # In app/services/transcription.py
   model_size = "base"  # tiny, base, small, medium, large-v3
   ```

2. **Enable GPU Mixed Precision**
   ```python
   # In app/core/config.py
   use_fp16 = True
   ```

3. **Optimize File Processing**
   ```python
   # Increase chunk size for longer audio files
   chunk_length_s = 60
   ```

## Security

### Data Privacy

- All audio files are processed locally
- Temporary files are automatically cleaned up
- No data is sent to external services
- Database contains only speaker traits (no audio data)

### Network Security

- HTTPS encryption in production
- Session-based authentication
- File upload validation
- Rate limiting for API endpoints

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support and questions:

- Create an issue on GitHub
- Check the [documentation](docs/)
- Review the [troubleshooting guide](#troubleshooting)

## Changelog

### v1.0.0 (2024-01-01)
- Initial release
- Basic transcription and diarization
- Web interface
- Speaker recognition
- Multiple export formats
- GPU acceleration support