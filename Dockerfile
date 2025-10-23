# Use Python 3.11 slim base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libffi-dev \
    libssl-dev \
    wget \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install NumPy 1.x FIRST (critical for spaCy compatibility)
RUN pip install --no-cache-dir "numpy==1.24.3"

# Install PyTorch CPU version compatible with NumPy 1.x
RUN pip install --no-cache-dir \
    "torch==2.3.1+cpu" \
    "torchaudio==2.3.1+cpu" \
    --index-url https://download.pytorch.org/whl/cpu

# Copy requirements files
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies in correct order
RUN pip install --no-cache-dir \
    -r requirements-dev.txt

# Download spaCy model with NumPy 1.x environment
RUN python -m spacy download en_core_web_sm

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Health check
RUN python -c "import numpy, torch, spacy; print(f'NumPy: {numpy.__version__}, PyTorch: {torch.__version__}, spaCy: {spacy.__version__}')"

# Run the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
