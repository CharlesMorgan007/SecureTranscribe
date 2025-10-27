"""
Main FastAPI application for SecureTranscribe.
Provides the web interface and API endpoints for audio transcription and diarization.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.status import (
    HTTP_401_UNAUTHORIZED,
    HTTP_404_NOT_FOUND,
    HTTP_429_TOO_MANY_REQUESTS,
)
import uvicorn

# Add app to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.config import get_settings, SECURITY_SETTINGS
from app.core.database import init_database, get_database
from app.services.queue_service import get_queue_service
from app.utils.exceptions import SecureTranscribeError, ValidationError, FileUploadError
from app.api.transcription import router as transcription_router
from app.api.speakers import router as speakers_router
from app.api.sessions import router as sessions_router
from app.api.queue import router as queue_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/securetranscribe.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting SecureTranscribe application...")

    # Ensure required directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs(settings.upload_dir, exist_ok=True)
    os.makedirs(settings.processed_dir, exist_ok=True)

    # Initialize database
    try:
        init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    # Start queue service
    try:
        queue_service = get_queue_service()
        queue_service.start()
        logger.info("Queue service started successfully")
    except Exception as e:
        logger.error(f"Failed to start queue service: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down SecureTranscribe application...")

    # Stop queue service
    try:
        queue_service = get_queue_service()
        queue_service.stop()
        logger.info("Queue service stopped")
    except Exception as e:
        logger.error(f"Error stopping queue service: {e}")


# Create FastAPI application
app = FastAPI(
    title="SecureTranscribe",
    description="Secure, offline audio transcription and speaker diarization with GPU acceleration",
    version="1.0.0",
    lifespan=lifespan,
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    SessionMiddleware,
    secret_key=settings.secret_key,
    max_age=SECURITY_SETTINGS["session_timeout"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="app/templates")

# Include API routers
app.include_router(
    transcription_router, prefix="/api/transcription", tags=["transcription"]
)
app.include_router(speakers_router, prefix="/api/speakers", tags=["speakers"])
app.include_router(sessions_router, prefix="/api/sessions", tags=["sessions"])
app.include_router(queue_router, prefix="/api/queue", tags=["queue"])


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main application page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        with next(get_database()) as db:
            db.execute("SELECT 1")

        # Check queue service
        queue_service = get_queue_service()
        queue_status = queue_service.get_queue_status()

        return {
            "status": "healthy",
            "timestamp": settings.get_settings().created_at,
            "queue_service": queue_status,
            "version": "1.0.0",
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.get("/info")
async def app_info():
    """Application information."""
    return {
        "name": "SecureTranscribe",
        "version": "1.0.0",
        "description": "Secure, offline audio transcription and speaker diarization",
        "features": [
            "Offline processing",
            "GPU acceleration",
            "Speaker diarization",
            "Multiple export formats",
            "Session management",
            "Queue-based processing",
        ],
        "supported_formats": [".mp3", ".wav", ".m4a", ".flac", ".ogg"],
        "export_formats": ["pdf", "csv", "txt", "json"],
    }


@app.exception_handler(SecureTranscribeError)
async def secure_transcribe_exception_handler(
    request: Request, exc: SecureTranscribeError
):
    """Handle custom SecureTranscribe exceptions."""
    return JSONResponse(status_code=400, content=exc.to_dict())


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle validation errors."""
    return JSONResponse(status_code=422, content=exc.to_dict())


@app.exception_handler(FileUploadError)
async def file_upload_exception_handler(request: Request, exc: FileUploadError):
    """Handle file upload errors."""
    return JSONResponse(status_code=400, content=exc.to_dict())


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "HTTP_ERROR", "message": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred. Please try again later.",
        },
    )


# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple rate limiting middleware."""
    client_ip = request.client.host
    rate_limit_config = SECURITY_SETTINGS["rate_limit"]

    # This is a simplified rate limiter - in production, use Redis or similar
    # For now, we'll just log the request
    logger.debug(f"Request from {client_ip}: {request.method} {request.url}")

    response = await call_next(request)
    return response


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
