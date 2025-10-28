"""
Transcription API router for SecureTranscribe.
Handles audio file upload, transcription, diarization, and export endpoints.
"""

import logging
import os
import shutil
import tempfile
import sys
from typing import Optional, List, Dict, Any
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Form,
    BackgroundTasks,
    Request,
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.database import get_database_manager, get_database
from app.core.config import get_settings
from app.utils.exceptions import (
    SecureTranscribeError,
    FileUploadError,
    TranscriptionError,
    ValidationError,
)
from app.models.transcription import Transcription
from app.models.session import UserSession
from app.models.speaker import Speaker
from app.models.processing_queue import ProcessingQueue
from app.services.queue_service import get_queue_service
from app.services.transcription_service import TranscriptionService
from app.services.diarization_service import DiarizationService
from app.services.export_service import ExportService
from app.services.audio_processor import AudioProcessor
from app.services.queue_service import get_queue_service

from app.utils.helpers import (
    ensure_directory_exists,
    safe_remove_file,
    format_file_size,
    format_duration,
    sanitize_filename,
    generate_unique_id,
)

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()


def get_current_session(
    request: Request, db: Session = Depends(get_database)
) -> UserSession:
    """Get or create user session."""
    session_token = request.session.get("session_token")

    if session_token:
        user_session = UserSession.get_by_token(db, session_token)
        if user_session and user_session.is_valid:
            user_session.update_last_accessed()
            return user_session

    # Create new session
    user_session = UserSession.create_session(
        db,
        user_agent=request.headers.get("user-agent"),
        ip_address=request.client.host,
    )

    request.session["session_token"] = user_session.session_token
    return user_session


@router.post("/upload")
async def upload_audio_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    auto_start: bool = Form(True),
    db: Session = Depends(get_database),
    user_session: UserSession = Depends(get_current_session),
):
    """
    Upload an audio file for transcription.

    Args:
        file: Audio file to upload
        language: Optional language code for transcription
        auto_start: Whether to automatically start processing
        db: Database session
        user_session: Current user session

    Returns:
        Upload result with file information
    """
    try:
        # Validate file
        if not file.filename:
            raise ValidationError("No file provided")

        # Sanitize filename
        safe_filename = sanitize_filename(file.filename)
        if not safe_filename:
            raise ValidationError("Invalid filename")

        # Check file extension
        allowed_extensions = [".mp3", ".wav", ".m4a", ".flac", ".ogg"]
        file_ext = os.path.splitext(safe_filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise ValidationError(f"Unsupported file format: {file_ext}")

        # Check file size
        file_content = await file.read()
        file_size = len(file_content)
        max_size = settings.max_file_size_bytes

        if file_size > max_size:
            raise ValidationError(
                f"File too large: {format_file_size(file_size)} "
                f"(max: {format_file_size(max_size)})"
            )

        # Generate unique filename
        unique_filename = f"{generate_unique_id()}_{safe_filename}"
        upload_path = os.path.join(settings.upload_dir, unique_filename)

        # Ensure upload directory exists
        ensure_directory_exists(settings.upload_dir)

        # Save file
        with open(upload_path, "wb") as f:
            f.write(file_content)

        # Validate audio file
        audio_processor = AudioProcessor()
        file_info = audio_processor.validate_file(upload_path)

        # Create transcription record
        transcription = Transcription(
            session_id=user_session.session_id,
            original_filename=safe_filename,
            file_path=upload_path,
            file_size=file_size,
            file_duration=file_info["duration"],
            file_format=file_info["file_format"],
            whisper_model=settings.whisper_model_size,
            pyannote_model=settings.pyannote_model,
        )

        db.add(transcription)
        db.commit()
        db.refresh(transcription)

        result = {
            "success": True,
            "transcription_id": transcription.id,
            "session_id": transcription.session_id,
            "filename": safe_filename,
            "file_size": file_size,
            "formatted_file_size": format_file_size(file_size),
            "duration": file_info["duration"],
            "formatted_duration": format_duration(file_info["duration"]),
            "format": file_info["file_format"],
            "language_detected": file_info.get("language"),
        }

        # Auto-start processing if requested
        if auto_start:
            background_tasks.add_task(
                start_transcription,
                transcription.id,
                user_session.session_id,
                language,
            )
            result["processing_started"] = True
        else:
            result["processing_started"] = False

        logger.info(f"File uploaded successfully: {safe_filename}")
        return result

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise FileUploadError(f"Failed to upload file: {str(e)}")


@router.post("/start/{transcription_id}")
async def start_transcription(
    transcription_id: int,
    language: Optional[str] = None,
    db: Session = Depends(get_database),
    user_session: UserSession = Depends(get_current_session),
):
    """
    Start transcription processing for an uploaded file.

    Args:
        transcription_id: ID of the transcription to start
        language: Optional language code
        db: Database session
        user_session: Current user session

    Returns:
        Processing start result
    """
    try:
        # Get transcription
        transcription = (
            db.query(Transcription)
            .filter(
                Transcription.id == transcription_id,
                Transcription.session_id == user_session.session_id,
            )
            .first()
        )

        if not transcription:
            raise HTTPException(status_code=404, detail="Transcription not found")

        if transcription.status != "pending":
            raise ValidationError(
                f"Cannot start transcription in '{transcription.status}' status"
            )

        # Submit to queue
        queue_service = get_queue_service()

        # Small delay to ensure transcription is committed to database
        # before queue processor picks it up
        time.sleep(0.1)  # 100ms delay

        job_id = queue_service.submit_job(
            session_id=user_session.session_id,
            file_path=transcription.file_path,
            file_size=transcription.file_size,
            file_duration=transcription.file_duration,
            transcription_id=transcription.id,
            priority=5,  # Default priority
            processing_options={"language": language} if language else {},
        )

        # Ensure database session is committed and refresh queue service
        # to prevent "Transcription not found" error
        db.commit()
        db.expire_all()

        # Update user session
        user_session.set_current_transcription(transcription.id)

        return {
            "success": True,
            "job_id": job_id,
            "transcription_id": transcription.id,
            "status": "queued",
            "message": "Transcription started and added to queue",
        }

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"Failed to start transcription: {e}")
        raise TranscriptionError(f"Failed to start transcription: {str(e)}")


@router.get("/status/{transcription_id}")
async def get_transcription_status(
    transcription_id: int,
    db: Session = Depends(get_database),
    user_session: UserSession = Depends(get_current_session),
):
    """
    Get the status of a transcription.

    Args:
        transcription_id: ID of the transcription
        db: Database session
        user_session: Current user session

    Returns:
        Transcription status information
    """
    try:
        transcription = (
            db.query(Transcription)
            .filter(
                Transcription.id == transcription_id,
                Transcription.session_id == user_session.session_id,
            )
            .first()
        )

        if not transcription:
            raise HTTPException(status_code=404, detail="Transcription not found")

        # Get queue position if queued
        queue_position = None
        estimated_wait_time = None

        if transcription.status == "pending":
            queue_service = get_queue_service()
            queue_position = queue_service.get_user_queue_position(
                user_session.session_id
            )
            estimated_wait_time = queue_service.estimate_wait_time(
                user_session.session_id
            )

        result = {
            "transcription_id": transcription.id,
            "session_id": transcription.session_id,
            "status": transcription.status,
            "progress_percentage": transcription.progress_percentage,
            "current_step": transcription.current_step,
            "created_at": transcription.created_at.isoformat()
            if transcription.created_at
            else None,
            "started_at": transcription.started_at.isoformat()
            if transcription.started_at
            else None,
            "completed_at": transcription.completed_at.isoformat()
            if transcription.completed_at
            else None,
            "processing_time": transcription.processing_time,
            "file_info": {
                "filename": transcription.original_filename,
                "duration": transcription.file_duration,
                "formatted_duration": transcription.formatted_duration,
                "format": transcription.file_format,
            },
            "queue_position": queue_position,
            "estimated_wait_time": estimated_wait_time,
            "error_message": transcription.error_message,
        }

        # Add result information if completed
        if transcription.is_completed:
            result.update(
                {
                    "full_transcript": transcription.full_transcript,
                    "language_detected": transcription.language_detected,
                    "confidence_score": transcription.confidence_score,
                    "num_speakers": transcription.num_speakers,
                    "speakers": transcription.get_speaker_list(),
                    "segments": transcription.segments,
                    "speaker_stats": transcription.get_speaker_stats(),
                    "preview_clips": transcription.generate_preview_clips(),
                }
            )

        return result

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"Failed to get transcription status: {e}")
        raise TranscriptionError(f"Failed to get transcription status: {str(e)}")


@router.post("/speakers/assign/{transcription_id}")
async def assign_speakers(
    transcription_id: int,
    speaker_assignments: Dict[str, str],
    db: Session = Depends(get_database),
    user_session: UserSession = Depends(get_current_session),
):
    """
    Assign names to identified speakers in a transcription.

    Args:
        transcription_id: ID of the transcription
        speaker_assignments: Dictionary mapping speaker labels to names
        db: Database session
        user_session: Current user session

    Returns:
        Assignment result
    """
    try:
        transcription = (
            db.query(Transcription)
            .filter(
                Transcription.id == transcription_id,
                Transcription.session_id == user_session.session_id,
            )
            .first()
        )

        if not transcription:
            raise HTTPException(status_code=404, detail="Transcription not found")

        if not transcription.is_completed:
            raise ValidationError(
                "Transcription must be completed before assigning speakers"
            )

        # Update speaker assignments
        for old_name, new_name in speaker_assignments.items():
            transcription.update_speaker_assignment(old_name, new_name)

            # Create or update speaker profile
            speaker = (
                db.query(Speaker)
                .filter(Speaker.name == new_name, Speaker.is_active == True)
                .first()
            )

            if not speaker:
                speaker = Speaker(name=new_name)
                db.add(speaker)
                db.commit()
                db.refresh(speaker)

            # Link transcription to speaker
            transcription.speaker_id = speaker.id

        db.commit()

        return {
            "success": True,
            "transcription_id": transcription.id,
            "speakers_assigned": len(speaker_assignments),
            "speakers": transcription.get_speaker_list(),
        }

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"Failed to assign speakers: {e}")
        raise ValidationError(f"Failed to assign speakers: {str(e)}")


@router.post("/export/{transcription_id}")
async def export_transcription(
    transcription_id: int,
    export_format: str,
    include_options: Optional[List[str]] = None,
    db: Session = Depends(get_database),
    user_session: UserSession = Depends(get_current_session),
):
    """
    Export transcription in specified format.

    Args:
        transcription_id: ID of the transcription
        export_format: Export format (pdf, csv, txt, json)
        include_options: Additional content to include
        db: Database session
        user_session: Current user session

    Returns:
        Exported file
    """
    try:
        transcription = (
            db.query(Transcription)
            .filter(
                Transcription.id == transcription_id,
                Transcription.session_id == user_session.session_id,
            )
            .first()
        )

        if not transcription:
            raise HTTPException(status_code=404, detail="Transcription not found")

        if not transcription.is_completed:
            raise ValidationError("Transcription must be completed before export")

        # Generate export
        export_service = ExportService()
        export_content = export_service.export_transcription(
            transcription, export_format, include_options, db
        )

        # Create temporary file
        temp_dir = tempfile.mkdtemp()
        filename = f"{transcription.original_filename}_transcript.{export_format}"
        temp_path = os.path.join(temp_dir, filename)

        with open(temp_path, "wb") as f:
            f.write(export_content)

        # Schedule cleanup
        import asyncio

        asyncio.create_task(cleanup_temp_file(temp_path))

        # Determine media type
        media_types = {
            "pdf": "application/pdf",
            "csv": "text/csv",
            "txt": "text/plain",
            "json": "application/json",
        }
        media_type = media_types.get(export_format, "application/octet-stream")

        return FileResponse(
            path=temp_path,
            filename=filename,
            media_type=media_type,
        )

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"Failed to export transcription: {e}")
        raise ExportError(f"Failed to export transcription: {str(e)}")


@router.delete("/{transcription_id}")
async def delete_transcription(
    transcription_id: int,
    db: Session = Depends(get_database),
    user_session: UserSession = Depends(get_current_session),
):
    """
    Delete a transcription and its associated files.

    Args:
        transcription_id: ID of the transcription to delete
        db: Database session
        user_session: Current user session

    Returns:
        Deletion result
    """
    try:
        transcription = (
            db.query(Transcription)
            .filter(
                Transcription.id == transcription_id,
                Transcription.session_id == user_session.session_id,
            )
            .first()
        )

        if not transcription:
            raise HTTPException(status_code=404, detail="Transcription not found")

        # Delete associated files
        if os.path.exists(transcription.file_path):
            safe_remove_file(transcription.file_path)

        # Cancel processing if in queue
        if transcription.status in ["pending", "processing"]:
            queue_service = get_queue_service()
            # Find and cancel job (simplified implementation)
            # Find job by transcription_id and cancel it
            with next(get_database()) as queue_db:
                job = (
                    queue_db.query(ProcessingQueue)
                    .filter(ProcessingQueue.transcription_id == transcription_id)
                    .filter(ProcessingQueue.status.in_(["queued", "processing"]))
                    .first()
                )
                if job:
                    queue_service.cancel_job(job.job_id)

        # Delete transcription record
        db.delete(transcription)
        db.commit()

        return {
            "success": True,
            "message": "Transcription deleted successfully",
            "transcription_id": transcription_id,
        }

    except SecureTranscribeError:
        raise
    except Exception as e:
        logger.error(f"Failed to delete transcription: {e}")
        raise TranscriptionError(f"Failed to delete transcription: {str(e)}")


@router.get("/list")
async def list_transcriptions(
    limit: int = 50,
    offset: int = 0,
    status_filter: Optional[str] = None,
    db: Session = Depends(get_database),
    user_session: UserSession = Depends(get_current_session),
):
    """
    List transcriptions for the current user session.

    Args:
        limit: Maximum number of results
        offset: Results offset
        status_filter: Filter by status
        db: Database session
        user_session: Current user session

    Returns:
        List of transcriptions
    """
    try:
        query = db.query(Transcription).filter(
            Transcription.session_id == user_session.session_id
        )

        if status_filter:
            query = query.filter(Transcription.status == status_filter)

        query = query.order_by(Transcription.created_at.desc())
        query = query.offset(offset).limit(limit)

        transcriptions = query.all()

        result = []
        for transcription in transcriptions:
            result.append(
                {
                    "id": transcription.id,
                    "session_id": transcription.session_id,
                    "original_filename": transcription.original_filename,
                    "status": transcription.status,
                    "progress_percentage": transcription.progress_percentage,
                    "file_duration": transcription.file_duration,
                    "formatted_duration": transcription.formatted_duration,
                    "file_format": transcription.file_format,
                    "num_speakers": transcription.num_speakers,
                    "confidence_score": transcription.confidence_score,
                    "created_at": transcription.created_at.isoformat()
                    if transcription.created_at
                    else None,
                    "completed_at": transcription.completed_at.isoformat()
                    if transcription.completed_at
                    else None,
                }
            )

        return {
            "transcriptions": result,
            "total": len(result),
            "limit": limit,
            "offset": offset,
        }

    except Exception as e:
        logger.error(f"Failed to list transcriptions: {e}")
        raise TranscriptionError(f"Failed to list transcriptions: {str(e)}")


# Helper function for cleanup
async def cleanup_temp_file(file_path: str):
    """Clean up temporary export file after some time."""
    import asyncio

    await asyncio.sleep(300)  # Wait 5 minutes
    safe_remove_file(file_path)
    if os.path.exists(os.path.dirname(file_path)):
        os.rmdir(os.path.dirname(file_path))


# Background task function
async def start_transcription(
    transcription_id: int, session_id: str, language: Optional[str]
):
    """Background task to start transcription processing."""
    try:
        queue_service = get_queue_service()

        with next(get_database()) as db:
            transcription = (
                db.query(Transcription)
                .filter(Transcription.id == transcription_id)
                .first()
            )
            if not transcription:
                logger.error(f"Transcription {transcription_id} not found")
                return

            job_id = queue_service.submit_job(
                session_id=session_id,
                file_path=transcription.file_path,
                file_size=transcription.file_size,
                file_duration=transcription.file_duration,
                transcription_id=transcription.id,
                priority=5,
                processing_options={"language": language} if language else {},
            )

            logger.info(f"Started transcription {transcription_id} with job {job_id}")

    except Exception as e:
        logger.error(f"Failed to start transcription in background: {e}")
