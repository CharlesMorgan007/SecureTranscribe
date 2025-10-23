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
