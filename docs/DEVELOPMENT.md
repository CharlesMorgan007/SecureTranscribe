# SecureTranscribe Development Guide

This guide provides comprehensive instructions for setting up, developing, testing, and contributing to the SecureTranscribe application.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Development Setup](#development-setup)
3. [Project Structure](#project-structure)
4. [Development Workflow](#development-workflow)
5. [Testing](#testing)
6. [Code Quality](#code-quality)
7. [Debugging](#debugging)
8. [Performance Testing](#performance-testing)
9. [Deployment](#deployment)
10. [Contributing Guidelines](#contributing-guidelines)

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows (WSL2 recommended for Windows)
- **Python**: 3.11 or higher
- **RAM**: Minimum 8GB, 16GB+ recommended for development
- **Storage**: At least 10GB free space
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)

### Software Dependencies

#### Required
- Python 3.11+
- Git
- Docker and Docker Compose (for containerized development)
- FFmpeg (for audio processing)
- Node.js 16+ (for frontend development, optional)

#### Optional (for GPU development)
- NVIDIA GPU drivers
- CUDA Toolkit 11.8+
- cuDNN 8+

#### Development Tools
- VS Code, PyCharm, or your preferred IDE
- Postman or similar API testing tool
- Database browser (DBeaver, pgAdmin, etc.)

### Hardware Requirements for Testing

**Development Machine (CPU-only)**:
- 8GB RAM minimum
- 2+ CPU cores
- 10GB free disk space

**Development Machine (with GPU)**:
- 16GB+ RAM
- NVIDIA GPU (RTX 3060 or better recommended)
- CUDA-compatible GPU drivers

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/SecureTranscribe.git
cd SecureTranscribe
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Using conda (optional)
conda create -n securetranscribe python=3.11
conda activate securetranscribe
```

### 3. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-asyncio pytest-cov black flake8 mypy pre-commit
```

### 4. Install System Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install ffmpeg libsndfile1 libsox-fmt-all sox curl git
```

#### macOS
```bash
brew install ffmpeg libsndfile sox
```

#### Windows (WSL2)
```bash
sudo apt update
sudo apt install ffmpeg libsndfile1 libsox-fmt-all sox curl git
```

### 5. Setup Pre-commit Hooks

```bash
pre-commit install
```

### 6. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit the .env file with your configuration
nano .env
```

Key environment variables for development:

```env
DEBUG=true
LOG_LEVEL=DEBUG
DATABASE_URL=sqlite:///./dev_securetranscribe.db
SECRET_KEY=dev-secret-key-for-development-only
USE_GPU=false  # Set to true if you have GPU
WHISPER_MODEL_SIZE=base  # Use smaller model for development
```

### 7. Initialize Database

```bash
python -m app.core.database init
```

### 8. Run Development Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The application will be available at `http://localhost:8000`

## Project Structure

```
SecureTranscribe/
├── app/                          # Main application package
│   ├── api/                      # FastAPI routers
│   │   ├── transcription.py      # Transcription endpoints
│   │   ├── speakers.py          # Speaker management endpoints
│   │   ├── sessions.py          # Session management endpoints
│   │   └── queue.py             # Queue management endpoints
│   ├── core/                     # Core application components
│   │   ├── config.py            # Configuration management
│   │   └── database.py          # Database setup and connection
│   ├── models/                   # Database models
│   │   ├── transcription.py      # Transcription model
│   │   ├── speaker.py           # Speaker model
│   │   ├── session.py           # User session model
│   │   └── processing_queue.py   # Queue model
│   ├── services/                 # Business logic services
│   │   ├── transcription_service.py  # Speech-to-text service
│   │   ├── diarization_service.py     # Speaker diarization service
│   │   ├── speaker_service.py         # Speaker management service
│   │   ├── export_service.py          # Export functionality
│   │   ├── queue_service.py           # Queue management
│   │   └── audio_processor.py          # Audio file processing
│   ├── static/                   # Static web assets
│   │   ├── css/
│   │   └── js/
│   ├── templates/                # HTML templates
│   ├── utils/                    # Utility functions
│   │   ├── exceptions.py        # Custom exceptions
│   │   └── helpers.py           # Helper functions
│   └── main.py                   # FastAPI application entry point
├── tests/                        # Test suite
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   ├── test_config.py           # Configuration tests
│   └── test_audio_processor.py  # Audio processing tests
├── docs/                        # Documentation
├── uploads/                     # Temporary upload storage
├── processed/                   # Processed files storage
├── logs/                        # Application logs
├── requirements.txt             # Python dependencies
├── pytest.ini                   # Pytest configuration
├── .pre-commit-config.yaml      # Pre-commit hooks
├── Dockerfile                   # Docker configuration
├── docker-compose.yml           # Docker Compose configuration
└── README.md                    # Project documentation
```

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Write code following the project's style guidelines
- Add tests for new functionality
- Update documentation as needed

### 3. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_config.py
```

### 4. Code Quality Checks

```bash
# Format code
black app/ tests/

# Lint code
flake8 app/ tests/

# Type checking
mypy app/

# Run all pre-commit hooks
pre-commit run --all-files
```

### 5. Commit Changes

```bash
git add .
git commit -m "feat: add your feature description"
```

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Create a pull request on GitHub with a detailed description of your changes.

## Testing

### Running Tests

#### Unit Tests
```bash
# Run all unit tests
pytest tests/unit/

# Run specific test
pytest tests/unit/test_config.py -v

# Run with coverage
pytest tests/unit/ --cov=app.unit --cov-report=html
```

#### Integration Tests
```bash
# Run all integration tests
pytest tests/integration/

# Run with specific markers
pytest -m "integration" -v
```

#### Test Coverage
```bash
# Generate coverage report
pytest --cov=app --cov-report=html --cov-report=term-missing

# View HTML report
open htmlcov/index.html
```

### Test Categories

- **Unit Tests**: Test individual functions and classes in isolation
- **Integration Tests**: Test interaction between components
- **API Tests**: Test REST API endpoints
- **Audio Processing Tests**: Test audio handling and processing
- **Database Tests**: Test database operations and models

### Writing Tests

#### Unit Test Example

```python
# tests/unit/test_config.py
import pytest
from app.core.config import Settings

def test_default_settings():
    settings = Settings()
    assert settings.database_url == "sqlite:///./securetranscribe.db"
    assert settings.debug is False

def test_whisper_model_validation():
    with pytest.raises(ValueError):
        Settings(whisper_model_size="invalid_model")
```

#### Integration Test Example

```python
# tests/integration/test_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
```

### Test Database

For testing, use a separate database:

```bash
# Set test database URL
export DATABASE_URL="sqlite:///./test_securetranscribe.db"

# Run tests
pytest
```

## Code Quality

### Code Formatting

We use Black for code formatting:

```bash
# Format all code
black app/ tests/

# Check formatting without making changes
black --check app/ tests/
```

### Linting

We use Flake8 for linting:

```bash
# Lint all code
flake8 app/ tests/

# Configuration in .flake8 file
```

### Type Checking

We use MyPy for type checking:

```bash
# Type check all code
mypy app/

# Configuration in mypy.ini or pyproject.toml
```

### Pre-commit Hooks

Pre-commit hooks run automatically before each commit:

```bash
# Install hooks
pre-commit install

# Run all hooks manually
pre-commit run --all-files
```

## Debugging

### Local Development

#### Enable Debug Mode
```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
```

#### Logging Configuration
```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

#### Debugging with IDE

**VS Code Debug Configuration** (`.vscode/launch.json`):
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: FastAPI",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/app/main.py",
            "console": "integratedTerminal",
            "env": {
                "DEBUG": "true",
                "LOG_LEVEL": "DEBUG"
            }
        }
    ]
}
```

### Common Debugging Scenarios

#### Audio Processing Issues
```python
# Add debug logging to audio processor
logger.debug(f"Processing file: {file_path}")
logger.debug(f"File info: {file_info}")
```

#### Database Issues
```python
# Enable SQL logging
from sqlalchemy.engine import Engine
from sqlalchemy import event

@event.listens_for(Engine, "before_cursor_execute")
def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    logger.debug(f"SQL: {statement}")
    logger.debug(f"Parameters: {parameters}")
```

#### GPU Issues
```python
import torch

logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
```

## Performance Testing

### Load Testing

Use tools like `locust` or `wrk` for load testing:

```bash
# Install locust
pip install locust

# Create locustfile.py
# Run load test
locust -f locustfile.py --host=http://localhost:8000
```

### Profiling

#### CPU Profiling
```python
import cProfile
import pstats

def profile_function():
    pr = cProfile.Profile()
    pr.enable()
    # Your code here
    pr.disable()
    
    stats = pstats.Stats(pr)
    stats.sort_stats('cumulative')
    stats.print_stats(10)
```

#### Memory Profiling
```bash
pip install memory-profiler

# Add decorator to functions
@profile
def memory_intensive_function():
    pass
```

### GPU Performance Monitoring

```python
import torch

# Monitor GPU memory
if torch.cuda.is_available():
    logger.info(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    logger.info(f"GPU Memory Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

## Deployment

### Development Deployment

```bash
# Using Docker Compose
docker-compose up -d

# Using Docker
docker build -t securetranscribe:dev .
docker run -p 8000:8000 securetranscribe:dev
```

### Production Deployment

#### Environment Setup
```bash
# Set production environment variables
export DEBUG=false
export LOG_LEVEL=INFO
export SECRET_KEY=your-production-secret-key
export DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

#### Docker Production
```bash
# Build production image
docker build -t securetranscribe:latest .

# Run with GPU support
docker-compose --profile gpu up -d

# Run with monitoring
docker-compose --profile monitoring up -d
```

#### Kubernetes Deployment
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/
```

## Contributing Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use Black for formatting (line length 88)
- Use descriptive variable and function names
- Add type hints to all functions
- Write comprehensive docstrings

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test changes
- `chore`: Maintenance changes

Examples:
```
feat(api): add speaker export endpoint

fix(audio): handle unsupported audio formats gracefully

docs(readme): update installation instructions
```

### Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Test** thoroughly
5. **Ensure** code quality checks pass
6. **Update** documentation
7. **Submit** a pull request with:
   - Clear description of changes
   - Testing instructions
   - Screenshots if applicable
   - Breaking changes highlighted

### Review Process

- All PRs require at least one review
- Automated tests must pass
- Code quality checks must pass
- Documentation must be updated
- Breaking changes must be clearly documented

### Release Process

1. Update version numbers
2. Update CHANGELOG.md
3. Create release tag
4. Build and publish Docker images
5. Update documentation

## Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Check model permissions
ls -la ~/.cache/huggingface/

# Clear model cache
rm -rf ~/.cache/huggingface/
```

#### Database Connection Issues
```bash
# Check database URL
echo $DATABASE_URL

# Test database connection
python -c "from app.core.database import get_database; next(get_database())"
```

#### GPU Issues
```bash
# Check GPU availability
nvidia-smi

# Check CUDA installation
nvcc --version

# Check PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"
```

### Getting Help

- Check the [GitHub Issues](https://github.com/yourusername/SecureTranscribe/issues)
- Review existing documentation
- Search for similar problems
- Create a new issue with detailed information

## Resources

### Documentation
- [API Documentation](http://localhost:8000/docs)
- [README.md](../README.md)
- [Configuration Guide](CONFIGURATION.md)

### Tools and Libraries
- [FastAPI](https://fastapi.tiangolo.com/)
- [Whisper](https://github.com/openai/whisper)
- [PyAnnote](https://github.com/pyannote/pyannote-audio)
- [SQLAlchemy](https://www.sqlalchemy.org/)

### Community
- [GitHub Discussions](https://github.com/yourusername/SecureTranscribe/discussions)
- [Discord Server](https://discord.gg/your-invite)

---

For questions or contributions, feel free to open an issue or submit a pull request!
```

You are an expert engineer and your task is to write a new file from scratch.

You MUST respond with the file content wrapped in triple backticks (```).
The backticks should be on their own line.
The text you output will be saved verbatim as the content of the content of the file.
Tool calls should be disabled.
Start your response with ```.

<file_path>
SecureTranscribe/docs/DEPLOYMENT.md
</file_path>

<edit_description>
Create comprehensive deployment documentation
</edit_description>
