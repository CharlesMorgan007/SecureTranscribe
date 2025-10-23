# SecureTranscribe Docker Configuration
# Multi-stage build for production deployment

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

# Set labels
LABEL maintainer="SecureTranscribe Team" \
    org.label-schema.build-date=$BUILD_DATE \
    org.label-schema.name="SecureTranscribe" \
    org.label-schema.description="Secure audio transcription and speaker diarization" \
    org.label-schema.url="https://github.com/yourusername/SecureTranscribe" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.vcs-url="https://github.com/yourusername/SecureTranscribe.git" \
    org.label-schema.vendor="SecureTranscribe" \
    org.label-schema.version=$VERSION \
    org.label-schema.schema-version="1.0"

# Set environment variables for build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    libsox-fmt-all \
    sox \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    SECURETRANSCRIBE_HOME=/app \
    UPLOAD_DIR=/app/uploads \
    PROCESSED_DIR=/app/processed \
    LOG_DIR=/app/logs

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    libsndfile1 \
    libsox-fmt-all \
    sox \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r securetranscribe && \
    useradd -r -g securetranscribe -d ${SECURETRANSCRIBE_HOME} -s /bin/bash securetranscribe

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create application directories
RUN mkdir -p ${SECURETRANSCRIBE_HOME} ${UPLOAD_DIR} ${PROCESSED_DIR} ${LOG_DIR} && \
    chown -R securetranscribe:securetranscribe ${SECURETRANSCRIBE_HOME} ${UPLOAD_DIR} ${PROCESSED_DIR} ${LOG_DIR}

# Set working directory
WORKDIR ${SECURETRANSCRIBE_HOME}

# Copy application code
COPY --chown=securetranscribe:securetranscribe . .

# Install application in development mode
RUN pip install -e .

# Create necessary directories and set permissions
RUN mkdir -p ${SECURETRANSCRIBE_HOME}/uploads ${SECURETRANSCRIBE_HOME}/processed ${SECURETRANSCRIBE_HOME}/logs && \
    chmod 755 ${SECURETRANSCRIBE_HOME}/uploads ${SECURETRANSCRIBE_HOME}/processed && \
    chmod 700 ${SECURETRANSCRIBE_HOME}/logs

# Copy entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Switch to non-root user
USER securetranscribe

# Expose port
EXPOSE 8000

# Set entrypoint
ENTRYPOINT ["docker-entrypoint.sh"]

# Set default command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# GPU-enabled variant
# To build with GPU support: docker build --target gpu -f Dockerfile .
FROM nvidia/cuda:11.8-runtime-ubuntu22.04 as gpu

# Install Python and system dependencies for GPU
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    python3.11-dev \
    curl \
    git \
    ffmpeg \
    libsndfile1 \
    libsox-fmt-all \
    sox \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
WORKDIR /app
COPY . .

# Install application
RUN pip install -e .

# Create directories
RUN mkdir -p /app/uploads /app/processed /app/logs

# Create non-root user
RUN groupadd -r securetranscribe && \
    useradd -r -g securetranscribe -d /app -s /bin/bash securetranscribe

# Set permissions
RUN chown -R securetranscribe:securetranscribe /app

# Switch to non-root user
USER securetranscribe

# Expose port
EXPOSE 8000

# Set CUDA environment
ENV CUDA_VISIBLE_DEVICES=0

# Set command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

You are an expert engineer and your task is to write a new file from scratch.

You MUST respond with the file's content wrapped in triple backticks (```).
The backticks should be on their own line.
The text you output will be saved verbatim as the content of the file.
Tool calls have been disabled.
Start your response with ```.

<file_path>
SecureTranscribe/docker-entrypoint.sh
</file_path>

<edit_description>
Create Docker entrypoint script for container initialization
</edit_description>
</think></think>
```bash
#!/bin/bash
# SecureTranscribe Docker Entrypoint Script
# Handles container initialization and startup

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a command exists
command_exists() {
command -v "$1" >/dev/null 2>&1
}

# Function to wait for database
wait_for_database() {
if [ -n "$DATABASE_URL" ]; then
log_info "Waiting for database connection..."

# Extract database type from URL
DB_TYPE=$(echo $DATABASE_URL | cut -d: -f1)

case $DB_TYPE in
"sqlite")
# For SQLite, just check if the file directory exists
DB_PATH=$(echo $DATABASE_URL | cut -d: -f3- | sed 's/\/\///')
DB_DIR=$(dirname "$DB_PATH")

if [ ! -d "$DB_DIR" ]; then
log_info "Creating SQLite database directory: $DB_DIR"
mkdir -p "$DB_DIR"
fi
;;
"postgresql")
# For PostgreSQL, wait for the server to be ready
if command_exists pg_isready; then
DB_HOST=$(echo $DATABASE_URL | cut -d@ -f2 | cut -d: -f1)
DB_PORT=$(echo $DATABASE_URL | cut -d@ -f2 | cut -d: -f2)

log_info "Waiting for PostgreSQL at $DB_HOST:$DB_PORT..."
while ! pg_isready -h "$DB_HOST" -p "$DB_PORT" -q; do
sleep 2
done
fi
;;
"mysql")
# For MySQL, wait for the server to be ready
if command_exists mysqladmin; then
DB_HOST=$(echo $DATABASE_URL | cut -d@ -f2 | cut -d: -f1)
DB_PORT=$(echo $DATABASE_URL | cut -d@ -f2 | cut -d: -f2)

log_info "Waiting for MySQL at $DB_HOST:$DB_PORT..."
while ! mysqladmin ping -h "$DB_HOST" -P "$DB_PORT" --silent; do
sleep 2
done
fi
;;
esac

log_success "Database is ready"
fi
}

# Function to initialize database
initialize_database() {
log_info "Initializing database..."

# Run database initialization
python -c "
from app.core.database import init_database
try:
init_database()
print('Database initialized successfully')
except Exception as e:
print(f'Database initialization failed: {e}')
exit(1)
"

if [ $? -eq 0 ]; then
log_success "Database initialized successfully"
else
log_error "Database initialization failed"
exit 1
fi
}

# Function to create necessary directories
create_directories() {
log_info "Creating necessary directories..."

directories=(
"$UPLOAD_DIR"
"$PROCESSED_DIR"
"$LOG_DIR"
"/tmp/securetranscribe"
)

for dir in "${directories[@]}"; do
if [ ! -d "$dir" ]; then
mkdir -p "$dir"
log_info "Created directory: $dir"
fi
done

# Set proper permissions
chmod 755 "$UPLOAD_DIR" "$PROCESSED_DIR"
chmod 700 "$LOG_DIR"

log_success "Directories created and permissions set"
}

# Function to check GPU availability
check_gpu() {
if [ "$USE_GPU" = "true" ] || [ "$USE_GPU" = "1" ]; then
log_info "GPU support requested, checking availability..."

if command_exists nvidia-smi; then
if nvidia-smi >/dev/null 2>&1; then
log_success "GPU detected and available"
export CUDA_VISIBLE_DEVICES=0
else
log_warning "GPU requested but nvidia-smi failed. Falling back to CPU."
export USE_GPU=false
fi
else
log_warning "GPU requested but nvidia-smi not found. Falling back to CPU."
export USE_GPU=false
fi
else
log_info "Running in CPU mode"
export USE_GPU=false
fi
}

# Function to download models if needed
download_models() {
log_info "Checking AI models..."

# This is a placeholder for model downloading logic
# In a real implementation, you might download models here if they're not present
python -c "
import os
from app.core.config import get_settings

settings = get_settings()
log_info(f'Using Whisper model: {settings.whisper_model_size}')
log_info(f'Using PyAnnote model: {settings.pyannote_model}')

# Check if models are available (this would depend on your specific setup)
log_info('AI models configuration verified')
"

if [ $? -eq 0 ]; then
log_success "AI models configuration verified"
else
log_error "AI models configuration failed"
exit 1
fi
}

# Function to setup logging
setup_logging() {
log_info "Setting up logging..."

# Create log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Set log file permissions
chmod 700 "$LOG_DIR"

# Create log file if it doesn't exist
LOG_FILE="$LOG_DIR/securetranscribe.log"
if [ ! -f "$LOG_FILE" ]; then
touch "$LOG_FILE"
fi

log_success "Logging setup completed"
}

# Function to run health checks
run_health_checks() {
log_info "Running pre-startup health checks..."

# Check if Python is working
python --version

# Check if required modules are available
python -c "
import sys
required_modules = [
'fastapi',
'uvicorn',
'sqlalchemy',
'librosa',
'soundfile'
]

missing_modules = []
for module in required_modules:
try:
__import__(module)
except ImportError:
missing_modules.append(module)

if missing_modules:
print(f'Missing required modules: {missing_modules}')
sys.exit(1)
else:
print('All required modules are available')
"

if [ $? -eq 0 ]; then
log_success "Health checks passed"
else
log_error "Health checks failed"
exit 1
fi
}

# Function to handle graceful shutdown
graceful_shutdown() {
log_info "Received shutdown signal, performing graceful shutdown..."

# Kill the main process
if [ -n "$MAIN_PID" ]; then
kill -TERM "$MAIN_PID" 2>/dev/null || true

# Wait for process to exit
wait "$MAIN_PID" 2>/dev/null || true
fi

log_success "Graceful shutdown completed"
exit 0
}

# Set up signal handlers
trap graceful_shutdown SIGTERM SIGINT

# Main execution
main() {
log_info "Starting SecureTranscribe container..."
log_info "Version: ${VERSION:-unknown}"
log_info "Build Date: ${BUILD_DATE:-unknown}"

# Print environment information
log_info "Python version: $(python --version)"
log_info "Working directory: $(pwd)"
log_info "User: $(whoami)"

# Run initialization steps
create_directories
setup_logging
check_gpu
wait_for_database
initialize_database
download_models
run_health_checks

# Set default environment variables if not set
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8000}
export DEBUG=${DEBUG:-false}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

log_info "Starting SecureTranscribe application..."
log_info "Host: $HOST"
log_info "Port: $PORT"
log_info "Debug: $DEBUG"
log_info "Log Level: $LOG_LEVEL"

# Start the application
if [ "$#" -eq 0 ]; then
# Default command
exec uvicorn app.main:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL" &
MAIN_PID=$!
else
# Custom command
exec "$@" &
MAIN_PID=$!
fi

# Wait for the main process
wait "$MAIN_PID"
}

# Check if script is being sourced or executed
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
main "$@"
fi
```

You are an expert engineer and your task is to write a file from scratch.

You MUST respond with the file content wrapped in triple backticks (```).
The backticks should be on their own line.
The text should be saved verbatim as the content of the file.
Tool calls should be disabled.
Start your response with ```.

<file_path>
SecureTranscribe/docker-compose.yml
</file_path>

<edit_description>
Create Docker Compose configuration for development and production
</edit_description>

```yaml
# SecureTranscribe Docker Compose Configuration
# Provides both development and production setups

version: '3.8'

services:
# Main application service
app:
build:
context: .
dockerfile: Dockerfile
target: production
args:
BUILD_DATE: ${BUILD_DATE:-$(date -u +'%Y-%m-%dT%H:%M:%SZ')}
VCS_REF: ${VCS_REF:-$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')}
VERSION: ${VERSION:-1.0.0}
container_name: securetranscribe-app
restart: unless-stopped
ports:
- "${PORT:-8000}:8000"
environment:
- DATABASE_URL=${DATABASE_URL:-sqlite:///./securetranscribe.db}
- SECRET_KEY=${SECRET_KEY:-dev-secret-key-change-in-production}
- DEBUG=${DEBUG:-false}
- LOG_LEVEL=${LOG_LEVEL:-INFO}
- HOST=0.0.0.0
- PORT=8000
- UPLOAD_DIR=/app/uploads
- PROCESSED_DIR=/app/processed
- USE_GPU=${USE_GPU:-false}
- WHISPER_MODEL_SIZE=${WHISPER_MODEL_SIZE:-base}
- PYANNOTE_MODEL=${PYANNOTE_MODEL:-pyannote/speaker-diarization-3.1}
- MAX_WORKERS=${MAX_WORKERS:-4}
- QUEUE_SIZE=${QUEUE_SIZE:-10}
- CLEANUP_DELAY=${CLEANUP_DELAY:-3600}
volumes:
- ./uploads:/app/uploads
- ./processed:/app/processed
- ./logs:/app/logs
- securetranscribe-data:/app/data
healthcheck:
test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
interval: 30s
timeout: 10s
retries: 3
start_period: 40s
networks:
- securetranscribe-network

# GPU-enabled variant (comment out if not using GPU)
app-gpu:
build:
context: .
dockerfile: Dockerfile
target: gpu
args:
BUILD_DATE: ${BUILD_DATE:-$(date -u +'%Y-%m-%dT%H:%M:%SZ')}
VCS_REF: ${VCS_REF:-$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')}
VERSION: ${VERSION:-1.0.0}
container_name: securetranscribe-app-gpu
restart: unless-stopped
ports:
- "${PORT:-8000}:8000"
environment:
- DATABASE_URL=${DATABASE_URL:-sqlite:///./securetranscribe.db}
- SECRET_KEY=${SECRET_KEY:-dev-secret-key-change-in-production}
- DEBUG=${DEBUG:-false}
- LOG_LEVEL=${LOG_LEVEL:-INFO}
- HOST=0.0.0.0
- PORT=8000
- UPLOAD_DIR=/app/uploads
- PROCESSED_DIR=/app/processed
- USE_GPU=true
- CUDA_VISIBLE_DEVICES=0
- WHISPER_MODEL_SIZE=${WHISPER_MODEL_SIZE:-base}
- PYANNOTE_MODEL=${PYANNOTE_MODEL:-pyannote/speaker-diarization-3.1}
- MAX_WORKERS=${MAX_WORKERS:-2}
- QUEUE_SIZE=${QUEUE_SIZE:-10}
volumes:
- ./uploads:/app/uploads
- ./processed:/app/processed
- ./logs:/app/logs
- securetranscribe-data:/app/data
deploy:
resources:
reservations:
devices:
- driver: nvidia
count: 1
capabilities: [gpu]
healthcheck:
test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
interval: 30s
timeout: 10s
retries: 3
start_period: 60s
networks:
- securetranscribe-network
profiles:
- gpu

# PostgreSQL database (optional)
postgres:
image: postgres:15-alpine
container_name: securetranscribe-postgres
restart: unless-stopped
environment:
- POSTGRES_DB=${POSTGRES_DB:-securetranscribe}
- POSTGRES_USER=${POSTGRES_USER:-securetranscribe}
- POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-securepassword}
- POSTGRES_INITDB_ARGS=--encoding=UTF-8 --lc-collate=C --lc-ctype=C
volumes:
- postgres-data:/var/lib/postgresql/data
- ./scripts/init-postgres.sql:/docker-entrypoint-initdb.d/init-postgres.sql:ro
ports:
- "${POSTGRES_PORT:-5432}:5432"
healthcheck:
test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-securetranscribe} -d ${POSTGRES_DB:-securetranscribe}"]
interval: 10s
timeout: 5s
retries: 5
networks:
- securetranscribe-network
profiles:
- database

# Redis for caching and session storage (optional)
redis:
image: redis:7-alpine
container_name: securetranscribe-redis
restart: unless-stopped
command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD:-redispassword}
volumes:
- redis-data:/data
ports:
- "${REDIS_PORT:-6379}:6379"
healthcheck:
test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
interval: 10s
timeout: 3s
retries: 5
networks:
- securetranscribe-network
profiles:
- cache

# Nginx reverse proxy (optional)
nginx:
image: nginx:alpine
container_name: securetranscribe-nginx
restart: unless-stopped
ports:
- "${HTTP_PORT:-80}:80"
- "${HTTPS_PORT:-443}:443"
volumes:
- ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
- ./nginx/ssl:/etc/nginx/ssl:ro
- ./uploads:/var/www/uploads:ro
- ./logs/nginx:/var/log/nginx
depends_on:
- app
healthcheck:
test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost/health"]
interval: 30s
timeout: 10s
retries: 3
networks:
- securetranscribe-network
profiles:
- production

# Monitoring with Prometheus (optional)
prometheus:
image: prom/prometheus:latest
container_name: securetranscribe-prometheus
restart: unless-stopped
ports:
- "${PROMETHEUS_PORT:-9090}:9090"
volumes:
- ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
- prometheus-data:/prometheus
command:
- '--config.file=/etc/prometheus/prometheus.yml'
- '--storage.tsdb.path=/prometheus'
- '--web.console.libraries=/etc/prometheus/console_libraries'
- '--web.console.templates=/etc/prometheus/consoles'
- '--storage.tsdb.retention.time=200h'
- '--web.enable-lifecycle'
networks:
- securetranscribe-network
profiles:
- monitoring

# Grafana for dashboards (optional)
grafana:
image: grafana/grafana:latest
container_name: securetranscribe-grafana
restart: unless-stopped
ports:
- "${GRAFANA_PORT:-3000}:3000"
environment:
- GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
- GF_USERS_ALLOW_SIGN_UP=false
volumes:
- grafana-data:/var/lib/grafana
- ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
- ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
depends_on:
- prometheus
networks:
- securetranscribe-network
profiles:
- monitoring

# Networks
networks:
securetranscribe-network:
driver: bridge
ipam:
config:
- subnet: 172.20.0.0/16

# Volumes
volumes:
securetranscribe-data:
driver: local
postgres-data:
driver: local
redis-data:
driver: local
prometheus-data:
driver: local
grafana-data:
driver: local

# Development override
x-dev-volumes: &dev-volumes
volumes:
- .:/app:delegated
- ./uploads:/app/uploads
- ./processed:/app/processed
- ./logs:/app/logs
- securetranscribe-data:/app/data

# Development services override
x-dev-environment: &dev-environment
environment:
- DEBUG=true
- LOG_LEVEL=DEBUG
- RELOAD=true

# Development configuration
# Usage: docker-compose -f docker-compose.yml -f docker-compose.override.yml up
# Create docker-compose.override.yml with:
# version: '3.8'
# services:
#   app:
#     <<: *dev-environment
#     <<: *dev-volumes
#     command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug
#     environment:
#       <<: *dev-environment
#       - DATABASE_URL=sqlite:///./dev_securetranscribe.db
```

You are an expert engineer and your task is to write a new file from scratch.

You MUST respond with the file content wrapped in triple backticks (```).
The backbacks should be on their own line.
The text you output will be saved verbatim as the content of the file.
Tool calls have been disabled.
Start your response with ``.

<file_path>
SecureTranscribe/.gitignore
</file_path>

<edit_description>
Create comprehensive .gitignore file for Python project
</edit_description>
</think></think>
```gitignore
# SecureTranscribe .gitignore
# Comprehensive ignore file for Python, AI models, and application data

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
.coverage.*
htmlcov/
.coverage.*
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Translations
*.mo
*.pot

# Django stuff:
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask stuff:
instance/
.webassets-cache

# Scrapy stuff:
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
#   According to pypa/pipenv#598, it is recommended to include Pipfile.lock in version control.
#   However, in case of collaboration, if having platform-specific dependencies or dependencies
#   having no cross-platform support, pipenv may install dependencies that don't work, or not
#   install all needed dependencies.
#Pipfile.lock

# PEP 582; used by e.g. github.com/David-OConnor/pyflow
__pypackages__/

# Celery stuff
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.env.local
.env.development
.env.test
.env.production
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# IDE - VSCode
.vscode/
*.code-workspace

# IDE - PyCharm
.idea/
*.iml
*.iws

# IDE - Eclipse
.project
.classpath
.cproject
.settings/

# macOS
.DS_Store
.AppleDouble
.LSOverride
Icon
._*
.DocumentRevisions-V100
.fseventsd
.Spotlight-V100
.TemporaryItems
.Trashes
.VolumeIcon.icns
.com.apple.timemachine.donotpresent
.AppleDB
.AppleDesktop
Network Trash Folder
Temporary Items
.apdisk

# Windows
Thumbs.db
Thumbs.db:encryptable
ehthumbs.db
ehthumbs_vista.db
*.stackdump
[Dd]esktop.ini
$RECYCLE.BIN/
*.lnk

# Linux
*~
.fuse_hidden*
.directory
.Trash-*
.nfs*

# Audio files and media
*.mp3
*.wav
*.m4a
*.flac
*.ogg
*.aac
*.wma
*.aiff
*.au
*.ra
*.amr
*.3gp
*.avi
*.mov
*.mp4
*.mkv
*.wmv
*.flv
*.webm

# SecureTranscribe specific
uploads/
processed/
temp/
tmp/
*.temp
*.tmp
logs/
*.log
securetranscribe.db
securetranscribe.db-journal
*.db
*.db-journal
data/
models/
cache/
sessions/
queue/

# AI/ML models and data
*.pt
*.pth
*.onnx
*.pb
*.model
*.bin
*.ckpt
*.safetensors
models_cache/
transformers_cache/
huggingface_cache/
torch_cache/
cache_dir/
.cache/

# Configuration files with secrets
config/secrets.yaml
config/local.yaml
config/production.yaml
secrets.yaml
local.yaml
production.yaml
*.key
*.pem
*.crt
*.p12
*.pfx

# Backup files
*.bak
*.backup
*.old
*.orig
*.rej
*~

# Editor temporary files
*.swp
*.swo
*~
.#*
\#*#
.#.gitignore
.gitignore~
.gitignore.bak

# OS generated files
.DS_Store?
ehthumbs.db
Icon?
Thumbs.db

# Node.js (if using for frontend)
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.npm
.eslintcache

# Frontend build artifacts
dist/
build/
out/

# Docker
.dockerignore
docker-compose.override.yml

# Kubernetes
*.kubeconfig

# Terraform
*.tfstate
*.tfstate.*
.terraform/
.terraform.lock.hcl

# Local development
dev/
dev-*
local/
local-*
test/
test-*
sandbox/
sandbox-*

# Performance and profiling
*.prof
*.profile
*.trace
performance/
profiling/

# Large files
*.zip
*.tar.gz
*.rar
*.7z
*.iso
*.dmg
*.pkg
*.exe
*.msi

# Certificates and keys
*.crt
*.key
*.pem
*.csr
*.p12
*.pfx
certificates/
keys/

# Database dumps
*.sql
*.dump
*.backup

# Monitoring and metrics
metrics/
monitoring/
prometheus-data/
grafana-data/

# Temporary and scratch files
scratch/
scratchpad/
notes/
TODO.md
NOTES.md

# Generated documentation
docs/_build/
docs/build/
site/

# PyTorch specific
*.pth
runs/
wandb/
mlruns/

# Hugging Face specific
~/.cache/huggingface/
hub/

# CUDA specific
*.cu
*.cuh
*.ptx

# Audio processing specific
audio_cache/
soundfile_cache/
librosa_cache/

# File watchers and watchers
.watcher/
.watchmanconfig

# Local environment scripts
scripts/local/
scripts/dev/
scripts/test/

# Custom application data
app_data/
user_data/
session_data/

# Backup and archive
backups/
archives/
old/

# Experimental features
experimental/
prototype/
sandbox/

# Documentation drafts
drafts/
unpublished/

# Third-party integrations
integrations/
third_party/

# License and legal
LICENSE.backup
LEGAL.md.backup

# Project management
.project
.settings/

# Misc
*.tmp
*.temp
*.swp
*.swo
*~
